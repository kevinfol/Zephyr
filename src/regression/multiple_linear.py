import numpy as np
from scipy.optimize import lsq_linear
from onnx.helper import make_node, make_tensor, make_tensor_value_info, make_graph
from onnx import TensorProto, GraphProto, numpy_helper
from . import GenericRegressor


class MultipleLinearRegression(GenericRegressor):
    """Regressor that solves the class Ordinary Least Squares problem
    subject to monotonicity constraints. The resulting model has 2
    parameters (coef_ and intercept_)
    """

    PARAM_GRID = {"": [None]}

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer.By default, there is no monotonic constraint"""
        GenericRegressor.__init__(self, *args, **kwargs)
        self.monotonic = False
        return

    def set_monotonic(self, monotone_constraints: list[bool]) -> None:
        """Sets up any monotonic constraints

        Args:
            monotone_constraints (list[bool]): a list with True/False elements,
                where each element indicates whether the corresponding predictor
                should have a monotonic relationship with the target.
        """
        self.monotonic = monotone_constraints

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute the model coefficients and intercept that minimize the
        least squares error.

        Args:
            X (np.ndarray): feature data of shape NxM
            y (np.ndarray): target data (must be of shape: Nx1)
        """

        # set monotonicity constraints
        if isinstance(self.monotonic, bool):
            monotone_list = [self.monotonic for i in range(X.shape[1])]
        elif isinstance(self.monotonic, list):
            monotone_list = self.monotonic
        else:
            monotone_list = [False for i in range(X.shape[1])]

        # set the monotonic bounds
        bounds = ([0.0 if m else -np.inf for m in monotone_list], np.inf)

        # add no bound for intercept
        bounds[0].append(-np.inf)

        # Create a one's array to find the model intercept
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        # Solve the linear least squares problem
        result = lsq_linear(
            A=X,
            b=y,
            bounds=bounds,
            method="trf",
            tol=1e-10,
        )

        # store the coefficients and intercept
        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]

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
            name="multiple_linear_regression",
            inputs=inputs,
            outputs=outputs,
            initializer=constants,
        )

        return graph
