import numpy as np
from collections.abc import Callable

# A list of all the scorers in this module.
ALL_SCORERS = ["rmse", "d2_rmse", "adj_rsqrd", "rsqrd"]


def calc_rmse(predicted: np.ndarray, observed: np.ndarray, *args, **kwargs) -> float:
    """Computes the root mean squared error of the predictions versus
    the observations.

    Args:
        predicted (np.ndarray): predicted values from the estimator. Must be a
            1-dimensional numpy ndarray
        observed (np.ndarray): actual observed values. Must be a 1-dimensional
            numpy ndarray

    Returns:
        float: root mean squared error
    """

    return np.sqrt(np.mean((predicted - observed) ** 2))


def calc_rmse_dsqrd(
    predicted: np.ndarray, observed: np.ndarray, *args, **kwargs
) -> float:
    """Computes the D2 score associated with the RMSE. This is useful because the
    D2 score can is more analgous to the R2 score and can be used to compared
    models with difference numbers of years. It is also a score where higher
    values  ==  more skill.

    see https://scikit-learn.org/stable/modules/model_evaluation.html#d2-score

    Args:
        predicted (np.ndarray): predicted values from the estimator. Must be a
            1-dimensional numpy ndarray
        observed (np.ndarray): actual observed values. Must be a 1-dimensional
            numpy ndarray

    Returns:
        float: RMSE D-squared score.
    """

    # calculate the RMSE between the target median and observations as a baseline
    median = np.median(observed)
    baseline_rmse = calc_rmse(np.full_like(observed, median), observed)

    # calculate the real RMSE
    rmse = calc_rmse(predicted, observed)

    return 1 - (rmse / baseline_rmse)


def calc_rsqrd(predicted: np.ndarray, observed: np.ndarray, *args, **kwargs) -> float:
    """Computes the unadjusted R2 value (coefficient of determination) given a
    set of observations and a set of forecast predictions.

    Args:
        predicted (np.ndarray): predicted values from the estimator. Must be a
            1-dimensional numpy ndarray
        observed (np.ndarray): actual observed values. Must be a 1-dimensional
            numpy ndarray

    Returns:
        float: unadjusted R2 value
    """

    mean = np.mean(observed)
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - mean) ** 2)

    return 1 - (ss_res / ss_tot)


def calc_adj_rsqrd(
    predicted: np.ndarray, observed: np.ndarray, n_feats: int, *args, **kwargs
) -> float:
    """Computes the adjusted R2 value given a set of observations and forecast
    predictions, as well as the number of predictor features

    Args:
        predicted (np.ndarray): predicted values from the estimator. Must be a
            1-dimensional numpy ndarray
        observed (np.ndarray): actual observed values. Must be a 1-dimensional
            numpy ndarray
        n_feats (int): Number of independent forecast predictors

    Returns:
        float: adjusted r2 value
    """

    r2 = calc_rsqrd(predicted, observed)
    n = len(predicted)
    return 1 - (1 - r2) * (n - 1) / (n - n_feats - 1)


def score_compare(score_1: list[float], score_2: list[float]) -> bool:
    """Compares 2 lists of scores.

    example:
    >>> score_compare([0.3], [0.2])
    True
    >>> score_compare([0.11, 0.24], [-0.04, 0.12])
    True
    >>> score_compare([0.45, 0.24], [0.25, 0.55])
    False

    Args:
        score_1 (list[float]): a list of scores (e.g. [r2=0.44, d2_rmse=0.53])
        score_2 (list[float]): a list of scores (e.g. [r2=0.13, d2_rmse=0.44])

    Returns:
        bool: True if score_1 is definitely better than score_2, else false.
    """
    return all([s1 >= s2 for s1, s2 in zip(score_1, score_2)])


def get(scorer_name: str = "adj_rsqrd") -> Callable:
    """_summary_

    Args:
        scorer_name (str, optional): name of the scorer to retrieve. Valid options are
            'rsqrd', 'adj_rsqrd', 'rmse'. Defaults to "adj_rsqrd".
            If multiple scorers are selected (e.g. by calling: 'adj_rsqrd+d1_rmse'),
            both scorer objects are returned

    Returns:
        list[Callable]: scorer functions
    """
    if scorer_name == "adj_rsqrd":
        return [calc_adj_rsqrd]
    if scorer_name == "rsqrd":
        return [calc_rsqrd]
    if scorer_name == "rmse":
        return [calc_rmse]
    if scorer_name == "d2_rmse":
        return [calc_rmse_dsqrd]
    if "+" in scorer_name:
        scorers = scorer_name.split("+")
        out = []
        for scorer in scorers:
            out.append(get(scorer)[0])
        return out
