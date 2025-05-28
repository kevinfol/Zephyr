import numpy as np
from onnx.helper import make_tensor, make_tensor_value_info, make_node, make_graph
from onnx import TensorProto, GraphProto
from src.scoring import calc_rmse
from scipy.stats import boxcox, t


def create_uncertainty_onnx_graph(
    predictions: np.ndarray, observations: np.ndarray, deg_freedom: int
) -> GraphProto:
    """Creates the ONNX graph object that generates both the normal uncertainty
    interval and the box-cox transformed uncertainty interval. Specifically,
    this function uses the predictions and observations to generate a student-t
    based lookup table of RMSE×T-value which when added to the forecast, will
    create a 1% to 99% uncertainty interval. For BoxCox, the same table is
    created but uses the BoxCox transformed RMSE instead.

    Note: If the bc_lambda value is really close to zero, the boxcox transformation
    is converted into a standard log-transformation

    Note #2: If there are negative values in the predictions or observations, we
    cannot create a BC/Log interval. BC/Log is only defined for positive values.
    That interval will be set to all-NaN if it is the case that there are negatives.

    Args:
        predictions (np.ndarray): generally the cross-validated predictions from the model
        observations (np.ndarray): the actual observations
        deg_freedom (int): the number of independent predictors in the model
            e.g. if a model takes 2 inputs, then this is '2'

    Returns:
        GraphProto: ONNX GraphProto object
    """

    # Compute the normal RMSE
    rmse = calc_rmse(predicted=predictions, observed=observations)

    # Create a 1% to 99% t-table
    t_table = [t.ppf(exc, df=deg_freedom) for exc in np.linspace(0.01, 0.99, 99)]

    # Check for negatives in observed or predictions
    # if there are negatives, the box-cox / log transforms wont work
    # and we should just stick to the normal formulation for uncertainty
    is_negative_values = False
    min_value = min(min(observations), min(predictions))
    if min_value <= 0:
        is_negative_values = True

    if is_negative_values:
        boxcox_lambda = np.nan
        boxcox_log_rmse = np.nan
    else:
        #  Compute the Box-Cox transformed RMSE
        try:
            boxcox_predictions, boxcox_lambda = boxcox(x=predictions.astype(np.float64))
            boxcox_observations = boxcox(
                x=observations.astype(np.float64), lmbda=boxcox_lambda
            )
            boxcox_log_rmse = calc_rmse(
                predicted=boxcox_predictions.astype(np.float32),
                observed=boxcox_observations.astype(np.float32),
            )

            # Check whether the min bc prediction will be nan
            min_prediction = (boxcox_lambda * min(boxcox_predictions) + 1) ** (
                1 / boxcox_lambda
            )
            if np.isnan(min_prediction):
                boxcox_lambda = np.nan
                boxcox_log_rmse = np.nan
        except:
            boxcox_lambda = np.nan
            boxcox_log_rmse = np.nan

        # Check if we should do a log-transform instead of a BC transform
        if abs(boxcox_lambda) < 1e-3:
            log_predictions = np.log(predictions)
            log_observations = np.log(observations)
            boxcox_log_rmse = calc_rmse(
                predicted=log_predictions, observed=log_observations
            )

    # Initialize the ONNX Pipeline
    bc_nodes = []
    log_nodes = []
    neg_nodes = []
    nodes = []
    inputs = []
    outputs = []
    bc_constants = []
    log_constants = []
    neg_constants = []
    constants = []

    # ONNX constants/initializers
    bc_constants.extend(
        [
            make_tensor(
                name="boxcox_or_log_rmse_t_table",
                data_type=TensorProto.FLOAT,
                dims=[99],
                vals=[boxcox_log_rmse * t_val for t_val in t_table],
            ),
            # BoxCox Lambda value
            make_tensor(
                name="boxcox_lambda",
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[np.float32(boxcox_lambda)],
            ),
            # Inverse BoxCox Lambda
            make_tensor(
                name="inverse_boxcox_lambda",
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[1 / np.float32(boxcox_lambda)],
            ),
            # Constant value of '1.0'
            make_tensor(
                name="one",
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[1.0],
            ),
        ]
    )
    log_constants.extend(
        [
            make_tensor(
                name="boxcox_or_log_rmse_t_table",
                data_type=TensorProto.FLOAT,
                dims=[99],
                vals=[boxcox_log_rmse * t_val for t_val in t_table],
            ),
        ]
    )
    neg_constants.extend(
        [
            make_tensor(
                name="neg_nan_table",
                data_type=TensorProto.FLOAT,
                dims=[99],
                vals=[np.nan for _ in t_table],
            ),
        ]
    )

    constants.extend(
        [
            # RMSE×T-value array
            make_tensor(
                name="rmse_t_table",
                data_type=TensorProto.FLOAT,
                dims=[99],
                vals=[rmse * t_val for t_val in t_table],
            ),
            # Constant of 1e-3
            make_tensor(
                name="one_e_neg_3",
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[1e-3],
            ),
            # Constant of Zero
            make_tensor(
                name="zero",
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[0.0],
            ),
            # BoxCox Lambda value
            make_tensor(
                name="boxcox_lambda",
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[np.float32(boxcox_lambda)],
            ),
        ]
    )

    # Inputs
    inputs.extend(
        [
            # Forecast output from estimator
            make_tensor_value_info(
                name="forecast_input", elem_type=TensorProto.FLOAT, shape=[None, 1]
            )
        ]
    )

    # Outputs
    outputs.extend(
        [
            # Regular/Normal based uncertainty interval
            make_tensor_value_info(
                name="forecast_uncertainty",
                elem_type=TensorProto.FLOAT,
                shape=[None, 99],
                doc_string="RMSE-based uncertainty. Ordered by exceedence value (1 to 99 percent)",
            ),
            # Boxcox based uncertainty interval
            make_tensor_value_info(
                name="forecast_uncertainty_boxcox_or_log",
                elem_type=TensorProto.FLOAT,
                shape=[None, 99],
                doc_string="Box-Cox (or log-transform) RMSE-based uncertainty. Ordered by exceedence value (1 to 99 percent). NOTE: can be NaN if there is a negative number issue.",
            ),
        ]
    )

    # Computation nodes
    bc_nodes.extend(
        [
            # Box Cox Uncertainty steps (see https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation)
            make_node(
                op_type="Pow",
                inputs=["forecast_input", "boxcox_lambda"],
                outputs=["bc_transform_step_1"],
                doc_string="Raise forecast to the lambda power: ([y^lambda] - 1) / lambda",
            ),
            make_node(
                op_type="Sub",
                inputs=["bc_transform_step_1", "one"],
                outputs=["bc_transform_step_2"],
                doc_string="Subtract 1: (y^lambda [- 1]) / lambda",
            ),
            make_node(
                op_type="Div",
                inputs=["bc_transform_step_2", "boxcox_lambda"],
                outputs=["bc_transform_step_3"],
                doc_string="Divide by lambda: (y^lambda - 1) / [lambda]",
            ),
            make_node(
                op_type="Add",
                inputs=["bc_transform_step_3", "boxcox_or_log_rmse_t_table"],
                outputs=["bc_uncertainties"],
                doc_string="Add the values: ZxRMSE(box-cox), where Z is the uncertainty value from student-T table.",
            ),
            make_node(
                op_type="Mul",
                inputs=["bc_uncertainties", "boxcox_lambda"],
                outputs=["bc_transform_step_4"],
                doc_string="Multiple by lambda: (([lambda * yBC]) + 1)^(1/lambda)",
            ),
            make_node(
                op_type="Add",
                inputs=["bc_transform_step_4", "one"],
                outputs=["bc_transform_step_5"],
                doc_string="Add 1: ((lambda * yBC) [+ 1])^(1/lambda)",
            ),
            make_node(
                op_type="Pow",
                inputs=["bc_transform_step_5", "inverse_boxcox_lambda"],
                outputs=["forecast_uncertainty_boxcox_or_log"],
                doc_string="Exponentiate to inverse lambda: ((lambda * yBC) + 1)[^(1/lambda)]",
            ),
        ]
    )
    boxcox_body = make_graph(
        name="boxcox_path",
        nodes=bc_nodes,
        inputs=[],
        outputs=[outputs[1]],
        initializer=bc_constants,
    )
    log_nodes.extend(
        [
            # Log transform uncertainty steps
            make_node(
                op_type="Log",
                inputs=["forecast_input"],
                outputs=["log_transform_forecast"],
                doc_string="Take the natural log of the forecast value.",
            ),
            make_node(
                op_type="Add",
                inputs=["log_transform_forecast", "boxcox_or_log_rmse_t_table"],
                outputs=["log_transform_uncertainty"],
                doc_string="Add the values: ZxRMSE(box-cox), where Z is the uncertainty value from student-T table.",
            ),
            make_node(
                op_type="Exp",
                inputs=["log_transform_uncertainty"],
                outputs=["forecast_uncertainty_boxcox_or_log"],
                doc_string="Exponentiate to get back to normal target-space.",
            ),
        ]
    )
    log_body = make_graph(
        name="log_path",
        nodes=log_nodes,
        inputs=[],
        outputs=[outputs[1]],
        initializer=log_constants,
    )
    neg_nodes.extend(
        [
            make_node(
                op_type="Add",
                inputs=["forecast_input", "neg_nan_table"],
                outputs=["forecast_uncertainty_boxcox_or_log"],
                doc_string="Negative value in predictions/observations/forecast. Cannot create this interval.",
            ),
        ]
    )
    neg_body = make_graph(
        name="neg_obs_pred_fcst_path",
        nodes=neg_nodes,
        inputs=[],
        outputs=[outputs[1]],
        initializer=neg_constants,
    )
    log_box_cox_nodes = [
        make_node(
            op_type="Abs",
            inputs=["boxcox_lambda"],
            outputs=["abs_bc_lambda"],
            doc_string="Check for box-cox lambda derivation, or natural log derivation.",
        ),
        make_node(
            op_type="LessOrEqual",
            inputs=["abs_bc_lambda", "one_e_neg_3"],
            outputs=["conditional"],
            doc_string="If lambda is less than 0.001, then we use a natural log instead of Box-Cox.",
        ),
        # Box cox OR Log uncertainty
        make_node(
            op_type="If",
            inputs=["conditional"],
            outputs=["forecast_uncertainty_boxcox_or_log"],
            then_branch=log_body,
            else_branch=boxcox_body,
        ),
    ]
    log_boxcox_body = make_graph(
        name="log_boxcox_path", nodes=log_box_cox_nodes, inputs=[], outputs=[outputs[1]]
    )
    nodes.extend(
        [
            # Normal Uncertainty (add forecast to RMSE×T-table)
            make_node(
                op_type="Add",
                inputs=["forecast_input", "rmse_t_table"],
                outputs=["forecast_uncertainty"],
                doc_string="Add normal T-Table based uncertainty interval",
            ),
            make_node(
                op_type="IsNaN",
                inputs=["boxcox_lambda"],
                outputs=["nan_conditional"],
                doc_string="If the BC lambda is NaN, then there's a negative number in the predictions or observations, and we can't compute BC uncertainty",
            ),
            make_node(
                op_type="LessOrEqual",
                inputs=["forecast_input", "zero"],
                outputs=["neg_forecast_conditional"],
                doc_string="If the forecast is negative, we cannot construct a BC/Log uncertainty.",
            ),
            make_node(
                op_type="Or",
                inputs=["nan_conditional", "neg_forecast_conditional"],
                outputs=["nan_or_neg_conditional"],
            ),
            make_node(
                op_type="If",
                inputs=["nan_or_neg_conditional"],
                outputs=["forecast_uncertainty_boxcox_or_log"],
                then_branch=neg_body,
                else_branch=log_boxcox_body,
                doc_string="If there is a negative in the predictions/observations/forecast, we cannot make a BC/Log uncertainty interval.",
            ),
        ]
    )

    graph = make_graph(
        name="uncertainty_generation",
        nodes=nodes,
        inputs=inputs,
        outputs=outputs,
        initializer=constants,
    )

    return graph
