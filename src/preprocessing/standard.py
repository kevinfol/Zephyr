import numpy as np
from onnx import TensorProto, GraphProto, numpy_helper
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info


class StandardPreprocessing:
    """A preprocessor class that transforms an input dataset into it's
    standardized values (i.e. a equivalent dataset that has zero-mean and
    unit standard deviation).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Class initializer."""
        return

    def fit(self, X: np.ndarray) -> None:
        """Computes the column-wise means and standard deviations of the input
        dataset and performs a standardization.

        Args:
            X (np.ndarray): Input dataset to convert to standardized version.
        """

        # Standardize the input data to zero-mean, unit variance
        self.means_ = np.mean(X, axis=0)
        self.stdev_ = np.std(X, axis=0)
        self.decomposed_data = (X - self.means_) / self.stdev_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms an input dataset into the standardized version of that
        dataset using the parameters found with the `fit` method.

        Args:
            X (np.ndarray): input dataset to be transformed into a standardized version

        Returns:
            np.ndarray: standardized version of the input dataset
        """

        return (X - self.means_) / self.stdev_

    def create_onnx_graph(self, input_names: list[str] = []) -> GraphProto:
        """Creates an ONNX graph representing the transformation of an input
        data stream into the standardized output.

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
                for i in range(self.means_.shape[0])
            ]
        )

        # Create graph output
        outputs.extend(
            [
                make_tensor_value_info(
                    "preprocessed_data", TensorProto.FLOAT, [None, self.means_.shape[0]]
                )
            ]
        )

        # Constants
        constants.extend(
            [
                numpy_helper.from_array(self.means_.astype(np.float32), name="means"),
                numpy_helper.from_array(self.stdev_.astype(np.float32), name="stdevs"),
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
                    ["predictor_data", "means"],
                    ["X_mu"],
                    doc_string="Subtract the feature-wise means (means) from the input data.",
                ),
                make_node(
                    "Div",
                    ["X_mu", "stdevs"],
                    ["preprocessed_data"],
                    doc_string="Divide by the feature-wise standard deviations (stdevs).",
                ),
            ]
        )

        # Build graph
        graph = make_graph(
            nodes=nodes,
            name="standardize",
            inputs=inputs,
            outputs=outputs,
            initializer=constants,
        )

        return graph
