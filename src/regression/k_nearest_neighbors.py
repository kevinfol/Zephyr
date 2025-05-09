import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from onnx.helper import make_graph, make_node, make_tensor_value_info, make_tensor
from onnx.compose import merge_graphs
from onnx import GraphProto, TensorProto
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from . import GenericRegressor


class NearestNeighborsRegression(GenericRegressor):
    """Regressor that uses nearest neighbors to make predictions. The prediction
    is determined using the 5 nearest analog years.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    for more information.
    """

    PARAM_GRID = {"n_neighbors": [3, 5]}

    def __init__(self, *args, **kwargs) -> None:
        """Class Constructor, no arguments are expected."""
        GenericRegressor.__init__(self, *args, **kwargs)
        self.regr = KNeighborsRegressor(algorithm="brute")

        return

    def set_monotonic(self, monotone_constraints: list[bool]) -> None:
        """UNUSED IN K-NEAREST NEIGHBORS. NOTE: IT IS POSSIBLE TO
        HAVE A MONOTONIC FUZZY NEAREST NEIGHBORS, BUT IT WOULD BE
        CUMBERSOME TO IMPLEMENT FROM SCRATCH.

        Args:
            monotone_constraints (list[bool]): a list with True/False elements,
                where each element indicates whether the corresponding predictor
                should have a monotonic relationship with the target.
        """
        self.monotonic = monotone_constraints

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Create the model parameters for the brute force model lookup table.

        Args:
            X (np.ndarray): feature data of shape NxM
            y (np.ndarray): target data (must be of shape: Nx1)
        """
        self.X = X
        self.regr.fit(X=X, y=y)

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

        return self.regr.predict(X=X)

    def create_onnx_graph(self) -> GraphProto:
        """Creates the ONNX graph that transforms an input layer containing
        new predictor data into a set of new responses/predictions using this
        regressor's fitted parameters.

        Returns:
            GraphProto: resulting ONNX graph
        """

        onnx_model = to_onnx(
            self.regr,
            name="k_nearest_neighbors",
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
            nodes=[make_node("Reshape", ["yIn", "newShape"], ["y"])],
            inputs=[make_tensor_value_info("yIn", TensorProto.FLOAT, [None, 1])],
            outputs=[make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            name="reshape",
            initializer=[make_tensor("newShape", TensorProto.INT64, [1], [-1])],
        )
        merged_graph = merge_graphs(graph, reshape_graph, io_map=[("y1", "yIn")])

        return merged_graph
