import numpy as np
from onnx import TensorProto, GraphProto, numpy_helper
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info


class NoPreprocessing:
    """A preprocessor class that does nothing. The transform method make no transformation
    and there are no fitted parameters. The ONNX graph just concatenates and
    reshapes the inputs
    """

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer."""
        return

    def fit(self, X: np.ndarray) -> None:
        """Only stores the dimensions of the input array.

        Args:
            X (np.ndarray): Input array
        """
        self.num_features = X.shape[1]

        return

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Performs no transformation

        Args:
            X (np.ndarray): input array to (not) transform

        Returns:
            np.ndarray: untransformed copy of input array
        """

        return X

    def create_onnx_graph(self, input_names: list[str] = []) -> GraphProto:
        """Creates an ONNX graph representing the concatenation of the input data.
        No other operations are performed.

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
                for i in range(self.num_features)
            ]
        )

        # Create graph output
        outputs.extend(
            [
                make_tensor_value_info(
                    "preprocessed_data", TensorProto.FLOAT, [None, self.num_features]
                )
            ]
        )

        # Constants
        # Nodes
        nodes.extend(
            [
                # Concatenate the input predictor data into one row and reshape it
                make_node(
                    "Concat",
                    [input_.name for input_ in inputs],
                    ["preprocessed_data"],
                    axis=1,
                    doc_string="Concatenate all the inputs into one array.",
                )
            ]
        )

        # Build graph
        graph = make_graph(
            nodes=nodes,
            name="no_op_preproceesor",
            inputs=inputs,
            outputs=outputs,
            initializer=constants,
        )

        return graph
