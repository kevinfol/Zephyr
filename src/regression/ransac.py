import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from onnx.helper import make_node, make_tensor, make_tensor_value_info, make_graph
from onnx import TensorProto, GraphProto, numpy_helper
from . import GenericRegressor


class RansacRegression(GenericRegressor):
    """RANSAC (RANdom SAmple Consensus) regression iteratively fits a random
    subsample of the X,Y data over and over to come to a consensus of the
    best fitted estimator of Y. In that way, it is more robust to outliers
    than OLS. See more here:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    """

    PARAM_GRID = {"": [None]}

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer. By default there is no monotonicity."""
        GenericRegressor.__init__(self, *args, **kwargs)
        # monotonicity
        self.monotonic = False

        # Regressor
        self.ransac_estimator = LinearRegression(positive=self.monotonic)
        self.regr = RANSACRegressor(estimator=self.ransac_estimator, min_samples=None)

        return

    def set_monotonic(self, monotone_constraints: list[bool]) -> None:
        """_summary_

        Args:
            monotone_constraints (list[bool]): a list with True/False elements,
                where each element indicates whether the corresponding predictor
                should have a monotonic relationship with the target.
        """
        self.monotonic = monotone_constraints
        if self.monotonic:
            self.ransac_estimator.set_params(**{"positive": True})
        else:
            self.ransac_estimator.set_params(**{"positive": False})

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

        self.X = X

        # Solve the OLS / L2-Norm problem
        self.regr.fit(X, y)

        # store the coefficients and intercept
        self.coef_ = self.regr.estimator_.coef_
        self.intercept_ = self.regr.estimator_.intercept_

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

        return self.regr.predict(X=X)

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
