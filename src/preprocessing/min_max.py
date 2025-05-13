import numpy as np
from onnx import TensorProto, GraphProto, numpy_helper
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info


class MinMaxPreprocessing:
    """A preprocessor class that transforms an input dataset into an equivalent
    dataset scaled to a maximum of 1 and a minimum of zero. Useful for Neural
    Nets.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer."""
        return

    def fit(self, X: np.ndarray) -> None:
        """Computes the column-wise mins and max's and scales.

        Args:
            X (np.ndarray): Input dataset to convert to min-max version.
        """

        # Standardize the input data to zero-mean, unit variance
        self.mins_ = np.min(X, axis=0)
        self.maxs_ = np.max(X, axis=0)
        self.decomposed_data = (X - self.mins_) / (self.maxs_ - self.mins_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms an input dataset into the min-max version using the parameters
        from the `fit` function.

        Args:
            X (np.ndarray): input dataset to be transformed into a minmax version

        Returns:
            np.ndarray: minmax version of the input dataset
        """

        return (X - self.mins_) / (self.maxs_ - self.mins_)

    def create_onnx_graph(self, input_names: list[str] = []) -> GraphProto:
        """Creates an ONNX graph representing the transformation of an input
        data stream into the minmax output.

        Args:
            input_names (list[str], optional): a list of input names to use
                instead of the default 'input_data_N' values. Useful for
                merging onnx graphs

        Returns:
            GraphProto: ONNX GraphProto object
        """

        # Initialize the ONNX Pipeline
        nodes = []
        inputs = []
        outputs = []
        constants = []

        # Create the graph inputs
        inputs.extend(
            [
                make_tensor_value_info(
                    f"data_input_{i}" if len(input_names) == 0 else input_names[i],
                    TensorProto.FLOAT,
                    [None, 1],
                )
                for i in range(self.mins_.shape[0])
            ]
        )

        # Create graph output
        outputs.extend(
            [
                make_tensor_value_info(
                    "preprocessed_data", TensorProto.FLOAT, [None, self.mins_.shape[0]]
                )
            ]
        )

        # Constants
        constants.extend(
            [
                numpy_helper.from_array(self.mins_.astype(np.float32), name="mins"),
                numpy_helper.from_array(
                    (self.maxs_ - self.mins_).astype(np.float32), name="divisor"
                ),
            ]
        )

        # Nodes
        nodes.extend(
            [
                # Concatenate the input predictor data into one row and reshape it
                make_node(
                    "Concat",
                    [input_.name for input_ in inputs],
                    ["predictor_data"],
                    axis=1,
                    doc_string="Concatenate all the inputs into one array.",
                ),
                # Standardize the input data
                make_node(
                    "Sub",
                    ["predictor_data", "mins"],
                    ["X_1"],
                    doc_string="Subtract the feature-wise minimums from the input data.",
                ),
                make_node(
                    "Div",
                    ["X_1", "divisor"],
                    ["preprocessed_data"],
                    doc_string="Divide by the feature-wise ranges (divisor).",
                ),
            ]
        )

        # Build graph
        graph = make_graph(
            nodes=nodes,
            name="minmax",
            inputs=inputs,
            outputs=outputs,
            initializer=constants,
        )

        return graph
