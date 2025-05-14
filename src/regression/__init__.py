class GenericRegressor:

    USE_PARALLEL_PROCESSING = False
    USE_THREADING = False

    class EmptyRegression:
        def set_params(self, *args, **kwargs):
            return

        def get_params(self, *args, **kwargs):
            return {}

    def __init__(self, *args, **kwargs):

        self.regr = self.EmptyRegression()

    def set_params(self, **params):
        valid_params = self.regr.get_params(deep=True)
        valid_keys = valid_params.keys()
        new_params = {key: value for key, value in params.items() if key in valid_keys}
        self.regr.set_params(**new_params)


from .multiple_linear import MultipleLinearRegression
from .k_nearest_neighbors import NearestNeighborsRegression
from .ridge import RidgeRegression
from .random_forest import RandomForestRegression
from .ransac import RansacRegression
from .neural_network import NeuralNetworkRegression


def get(regr_name: str) -> object:
    """Returns the requested regression algorithm object, according to the
    passed `name`

    Args:
        regr_name (str): Requested regression object. Can be one of 'multiple_linear',
            'k_nearest_neighbors', or 'ridge'

    Returns:
        object: Regression algorithm object.
    """

    if regr_name == "multiple_linear":
        return MultipleLinearRegression
    if regr_name == "k_nearest_neighbors":
        return NearestNeighborsRegression
    if regr_name == "ridge":
        return RidgeRegression
    if regr_name == "random_forest":
        return RandomForestRegression
    if regr_name == "ransac":
        return RansacRegression
    if regr_name == "neural_network":
        return NeuralNetworkRegression
