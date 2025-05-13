from .principal_components import PrincipalComponentsPreprocessing
from .standard import StandardPreprocessing
from .noop import NoPreprocessing
from .min_max import MinMaxPreprocessing


def get(name: str = "none") -> object:
    """Returns the preprocessor class object associated with the given preprocessor name.

    Args:
        name (str, optional): preprocessor name. Can be one of 'principal_components',
        'standard', or 'none'. Defaults to "none".

    Returns:
        object: preprocessor object that has the `fit`, `transform`, and `create_onnx_graph` methods.
    """
    if name == "principal_components":
        return PrincipalComponentsPreprocessing
    elif name == "standard":
        return StandardPreprocessing
    elif name == "none":
        return NoPreprocessing
    elif name == "minmax":
        return MinMaxPreprocessing
