import numpy as np
from onnx import TensorProto, GraphProto, numpy_helper
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info


class PrincipalComponentsPreprocessing:
    """A preprocessor class that transforms an input dataset into it's
    principal components values, and returns a certain number of components
    (either user-specified, or automatically generated).
    """

    def __init__(self, num_components: int = 0) -> None:
        """Class initializer.

        Args:
            num_components (int, optional): Specifies the number of principal
                components to return when the "transform" function is called.
                If '0' (default), the number of components returned will
                correspond to the number of components needed to explain 90% of
                the variance in the original dataset. (or X.shape[1] - 1, whichever
                is smaller)
                Defaults to 0.
        """
        self.num_components = num_components

    def fit(self, X: np.ndarray) -> None:
        """Computes the eigenvectors/eigenvalues required to construct the
        Principal components of the input dataset.

        Args:
            X (np.ndarray): Input dataset to convert to principal components.
        """

        # Standardize the input data to zero-mean, unit variance
        self.means_ = np.mean(X, axis=0)
        self.stdev_ = np.std(X, axis=0)
        standard_x = (X - self.means_) / self.stdev_

        # If there is only one feature in the input dataset, it's impossible
        # to convert to PCs, so dont actually do any fitting/transforming
        if standard_x.shape[1] == 1:
            self.num_components = 1
            self.decomposed_data = standard_x
            self.eigen_vectors = np.full(standard_x.shape, 1.0).T
            return

        # Compute the covariance matrix of the data
        covariance = np.cov(standard_x.T)

        # Compute the eigenvalues and eigenvectors
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(covariance)

        # Sort the eigenvalues by importance
        sort_index = np.argsort(self.eigen_values)
        self.eigen_values = self.eigen_values[sort_index]
        self.eigen_vectors = self.eigen_vectors[:, sort_index]

        # If num_components is in 'auto' mode, find the correct number
        # to explain 90% variance
        if self.num_components == 0:
            self.num_components = (
                np.where(
                    np.cumsum(self.eigen_values, axis=0) / np.sum(self.eigen_values)
                    >= 0.90
                )[0][0]
                + 1
            )
            self.num_components = min(self.num_components, standard_x.shape[1] - 1)

        # Decompose the input data
        self.decomposed_data = np.matmul(self.eigen_vectors.T, standard_x.T).T[
            :, : self.num_components
        ]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input array, X, into its decomposed
        principal components. Returns only the number of principal
        components specified in the class initializer.

        Args:
            X (np.ndarray): input array of predictor data to decompose

        Returns:
            np.ndarray: principal component data (shape is ROWS(X) x n_components)
        """
        x_standard = (X - self.means_) / self.stdev_
        return np.matmul(self.eigen_vectors.T, x_standard.T).T[:, : self.num_components]

    def create_onnx_graph(self, input_names: list[str] = []) -> GraphProto:
        """Creates an ONNX graph representing the transformation of an input
        data stream into the principal components output.

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
                    "preprocessed_data", TensorProto.FLOAT, [None, self.num_components]
                )
            ]
        )

        # Constants
        constants.extend(
            [
                numpy_helper.from_array(self.means_.astype(np.float32), name="means"),
                numpy_helper.from_array(self.stdev_.astype(np.float32), name="stdevs"),
                numpy_helper.from_array(
                    self.eigen_vectors.astype(np.float32), name="eigen_vectors"
                ),
                make_tensor(
                    "num_components",
                    TensorProto.INT64,
                    [self.num_components],
                    list(range(self.num_components)),
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
                    ["predictor_data", "means"],
                    ["X_mu"],
                    doc_string="Subtract the feature-wise means (means) from the input data.",
                ),
                make_node(
                    "Div",
                    ["X_mu", "stdevs"],
                    ["standardized_X"],
                    doc_string="Divide by the feature-wise standard deviations (stdevs).",
                ),
                # Tanspose the standardized data and multiply by eigenvectors
                make_node(
                    "Transpose",
                    ["standardized_X"],
                    ["standardized_Xt"],
                    perm=[1, 0],
                    doc_string="Transpose the standardized data before multiplying by eigenvectors.",
                ),
                make_node(
                    "MatMul",
                    ["eigen_vectors", "standardized_Xt"],
                    ["PC_all"],
                    doc_string="Multiply standardized data by eigenvectors.",
                ),
                # Transpose again and gather the correct number of components
                make_node("Transpose", ["PC_all"], ["PC_all_t"], perm=[1, 0]),
                make_node(
                    "Gather",
                    ["PC_all_t", "num_components"],
                    ["preprocessed_data"],
                    axis=1,
                    doc_string="Choose only the required number of principal components.",
                ),
            ]
        )

        # Build graph
        graph = make_graph(
            nodes=nodes,
            name="principal_components",
            inputs=inputs,
            outputs=outputs,
            initializer=constants,
        )

        return graph
