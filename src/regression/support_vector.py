import numpy as np
from sklearn.svm import SVR
from onnx.helper import make_graph, make_node, make_tensor_value_info, make_tensor
from onnx.compose import merge_graphs
from onnx import GraphProto, TensorProto
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from . import GenericRegressor


class SupportVectorRegression(GenericRegressor):
    """Regressor that uses the Epsilon-Support Vector Regression algorithm
    to fit a regression model between predictors and observations

    see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    for more information
    """

    PARAM_GRID = {"epsilon": [0.05, 0.1, 0.2], "C": [1.0, 2.0]}

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer. By default there is no monotonicity."""
        GenericRegressor.__init__(self, *args, **kwargs)
        # monotonicity
        self.monotonic = False

        # Regressor
        self.regr = SVR(kernel="rbf", gamma="scale")

        return

    def set_monotonic(self, monotone_constraints: list[bool]) -> None:
        """UNUSED IN SVR. NOTE: IT IS POSSIBLE TO
        HAVE A MONOTONIC SVR, BUT IT WOULD BE
        CUMBERSOME TO IMPLEMENT FROM SCRATCH.

        Args:
            monotone_constraints (list[bool]): a list with True/False elements,
                where each element indicates whether the corresponding predictor
                should have a monotonic relationship with the target.
        """
        self.monotonic = monotone_constraints

    def has_zero_importance_predictors(self) -> bool:
        """If any of the predictors have zero importance / effect on the output
        of the model (e.g. zero-coefficient), then this function returns True,
        otherwise it returns False. It is useful for filtering out models.

        Returns:
            bool: True if one or more predictors have no effect on the
                model output. Otherwise, False
        """
        # Try a sensitivity analysis on each predictor to determine importance
        means = np.mean(self.X, axis=0, keepdims=True)
        mins = np.min(self.X, axis=0)
        maxs = np.max(self.X, axis=0)
        for input_num in range(self.X.shape[1]):
            test = means.copy()
            test[0, input_num] = maxs[input_num]
            output = self.predict(test)
            test[0, input_num] = mins[input_num]
            output2 = self.predict(test)
            if abs(output - output2) < 1e-5:
                return True
        return False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit the model parameters.

        Args:
            X (np.ndarray): feature data of shape NxM
            y (np.ndarray): target data (must be of shape: Nx1)
        """

        # Note: for the epsilon / C parameters to work universally,
        # Y must be standardized/preprocessed
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        self.y_proc = (y - self.y_min) / (self.y_max - self.y_min)
        # self.y_proc = self.y_proc.unsqueeze(-1)

        self.X = X
        self.regr.fit(X=X, y=self.y_proc)

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts a new response (or responses) using the model parameters
        and the given new predictor data

        Args:
            X (np.ndarray): an array of shape NxM that contains new observations
                of predictor data to create predictions from

        Returns:
            np.ndarray: predictions created from the new predictor values
        """

        return (self.regr.predict(X=X) * (self.y_max - self.y_min)) + self.y_min

    def create_onnx_graph(self) -> GraphProto:
        """Creates the ONNX graph that transforms an input layer containing
        new predictor data into a set of new responses/predictions using this
        regressor's fitted parameters.

        Returns:
            GraphProto: resulting ONNX graph
        """

        onnx_model = to_onnx(
            self.regr,
            name="random_forest",
            initial_types=[
                (
                    "x",
                    FloatTensorType([None, self.X.shape[1]]),
                )
            ],
            final_types=[
                (
                    "y1",
                    FloatTensorType([None, 1]),
                )
            ],
            target_opset={"": 19, "ai.onnx.ml": 2},
            model_optim=True,
        )

        graph = onnx_model.graph
        reshape_graph = make_graph(
            nodes=[make_node("Reshape", ["yIn", "newShape"], ["y2"])],
            inputs=[make_tensor_value_info("yIn", TensorProto.FLOAT, [None, 1])],
            outputs=[make_tensor_value_info("y2", TensorProto.FLOAT, [None])],
            name="reshape",
            initializer=[make_tensor("newShape", TensorProto.INT64, [1], [-1])],
        )
        merged_graph = merge_graphs(graph, reshape_graph, io_map=[("y1", "yIn")])

        postprocess = make_graph(
            nodes=[
                make_node("Mul", ["yIn", "yStd"], ["post_1"]),
                make_node("Add", ["post_1", "yMean"], ["post_2"]),
                make_node("Reshape", ["post_2", "newShape2"], ["y"]),
            ],
            name="NeuralNetPostProcess",
            inputs=[make_tensor_value_info("yIn", TensorProto.FLOAT, [None, 1])],
            outputs=[make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            initializer=[
                make_tensor("newShape2", TensorProto.INT64, [1], [-1]),
                make_tensor("yMean", TensorProto.FLOAT, [1], [self.y_min]),
                make_tensor("yStd", TensorProto.FLOAT, [1], [self.y_max - self.y_min]),
            ],
        )
        merged_graph = merge_graphs(merged_graph, postprocess, io_map=[("y2", "yIn")])

        return merged_graph
