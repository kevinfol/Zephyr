import numpy as np
from sklearn.linear_model import Ridge
from onnx.helper import make_node, make_tensor, make_tensor_value_info, make_graph
from onnx import TensorProto, GraphProto, numpy_helper
from . import GenericRegressor


class RidgeRegression(GenericRegressor):
    """Regressor that solves the class Ordinary Least Squares problem
    subject to monotonicity constraints and a L2-Norm regularization constraint.
    The resulting model has 2 parameters (coef_ and intercept_).

    see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    for more information
    """

    PARAM_GRID = {"alpha": [0.5, 1.0, 1.5]}

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer. By default there is no monotonicity."""
        GenericRegressor.__init__(self, *args, **kwargs)
        # monotonicity
        self.monotonic = False

        # Regressor
        self.regr = Ridge(
            positive=self.monotonic,
            solver="lbfgs" if self.monotonic else "auto",
            tol=1e-3,
        )

        return

    def set_monotonic(self, monotone_constraints: list[bool]) -> None:
        """_summary_

        Args:
            monotone_constraints (list[bool]): a list with True/False elements,
                where each element indicates whether the corresponding predictor
                should have a monotonic relationship with the target.
        """
        self.monotonic = any(monotone_constraints)
        if self.monotonic:
            self.regr.set_params(**{"positive": True, "solver": "lbfgs"})
        else:
            self.regr.set_params(**{"positive": False, "solver": "auto"})

    def has_zero_importance_predictors(self) -> bool:
        """If any of the predictors have zero importance / effect on the output
        of the model (e.g. zero-coefficient), then this function returns True,
        otherwise it returns False. It is useful for filtering out models.

        Returns:
            bool: True if one or more predictors have no effect on the
                model output. Otherwise, False
        """
        if any(np.abs(self.coef_) < 1e-7):
            return True
        return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute the model coefficients and intercept that minimize the
        least squares error.

        Args:
            X (np.ndarray): feature data of shape NxM
            y (np.ndarray): target data (must be of shape: Nx1)
        """

        # Solve the OLS / L2-Norm problem
        self.regr.fit(X, y)

        # store the coefficients and intercept
        self.coef_ = self.regr.coef_
        self.intercept_ = self.regr.intercept_

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts a new response (or responses) using the model coefficents
        and the given new predictor data

        Args:
            X (np.ndarray): an array of shape NxM that contains new observations
                of predictor data to create predictions from

        Returns:
            np.ndarray: predictions created from the new predictor values
        """

        return np.matmul(X, self.coef_) + self.intercept_

    def create_onnx_graph(self) -> GraphProto:
        """Creates the ONNX graph that transforms an input layer containing
        new predictor data into a set of new responses/predictions using this
        regressor's fitted parameters.

        Returns:
            GraphProto: resulting ONNX graph
        """

        # Initialize ONNX graph
        inputs = []
        outputs = []
        nodes = []
        constants = []

        # Inputs
        inputs.extend(
            [make_tensor_value_info("x", TensorProto.FLOAT, [None, len(self.coef_)])]
        )

        # Outputs
        outputs.extend([make_tensor_value_info("y", TensorProto.FLOAT, [None])])

        # Constants
        constants.extend(
            [
                numpy_helper.from_array(
                    self.coef_.astype(np.float32), name="coefficients"
                ),
                make_tensor("intercept", TensorProto.FLOAT, [1], [self.intercept_]),
            ]
        )

        # Nodes
        nodes.extend(
            [
                make_node("MatMul", ["x", "coefficients"], ["y_no_intercept"]),
                make_node("Add", ["y_no_intercept", "intercept"], ["y"]),
            ]
        )

        # Make graph
        graph = make_graph(
            nodes=nodes,
            name="ridge_regression",
            inputs=inputs,
            outputs=outputs,
            initializer=constants,
        )

        return graph
