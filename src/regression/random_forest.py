import numpy as np
from sklearn.ensemble import RandomForestRegressor
from onnx.helper import make_graph, make_node, make_tensor_value_info, make_tensor
from onnx.compose import merge_graphs
from onnx import GraphProto, TensorProto
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from . import GenericRegressor


class RandomForestRegression(GenericRegressor):
    """ """

    PARAM_GRID = {
        "max_depth": [
            7,
            4,
        ],
        "min_samples_split": [0.075, 0.125],
    }

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer. By default there are no monotonic constraints."""
        GenericRegressor.__init__(self, *args, **kwargs)
        # monotonicity
        self.monotonic = False

        self.regr = RandomForestRegressor(
            n_estimators=35, max_features="log2", min_samples_leaf=0.035
        )

        return

    def set_monotonic(self, monotone_constraints: list[bool]) -> None:
        """_summary_

        Args:
            monotone_constraints (list[bool]): a list with True/False elements,
                where each element indicates whether the corresponding predictor
                should have a monotonic relationship with the target.
        """
        self.monotonic = monotone_constraints
        self.regr.set_params(**{"monotonic_cst": [int(m) for m in self.monotonic]})

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
            nodes=[make_node("Reshape", ["yIn", "newShape"], ["y"])],
            inputs=[make_tensor_value_info("yIn", TensorProto.FLOAT, [None, 1])],
            outputs=[make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            name="reshape",
            initializer=[make_tensor("newShape", TensorProto.INT64, [1], [-1])],
        )
        merged_graph = merge_graphs(graph, reshape_graph, io_map=[("y1", "yIn")])

        return merged_graph
