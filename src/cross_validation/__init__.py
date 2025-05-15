from collections.abc import Callable
import random


def k_fold(num_samples: int, num_folds: int = 5) -> list[list, list]:
    """Creates training and testing sets using a KFold strategy.
    For example:
        k_fold(6, 3) ->
        [
            # Testing set        Training set
            [ [0,1],             [2,3,4,5]       ],
            [ [2,3],             [0,1,4,5]       ],
            [ [4,5],             [0,1,2,3]       ]
        ]

    Args:
        num_samples (int): total number of samples to split up
        num_folds (int, optional): number of folds in cross-validation.
            If set to `num_samples`, this algorithm is equivalent
            to leave-one-out cross validation. Defaults to 5.

    Returns:
        list[list, list]: A list where each entry contains a test set, and a training set
            in that order.
    """

    # Determine the number of samples in each fold
    min_num_samples_per_fold = num_samples // num_folds
    remainder = num_samples % num_folds
    num_samples_per_fold = [min_num_samples_per_fold] * num_folds
    for i in range(remainder):
        num_samples_per_fold[i] += 1

    # List of available indices
    available_indices = list(range(num_samples))

    # shuffle indices
    random.shuffle(available_indices)

    # create the train/test samples
    output = []
    current_idx = 0
    for fold in range(num_folds):
        end_idx = current_idx + num_samples_per_fold[fold]
        test_group = available_indices[current_idx:end_idx]
        train_group = list(set(available_indices).difference(test_group))
        output.append([test_group, train_group])
        current_idx += num_samples_per_fold[fold]

    return output


def k_fold_5(num_samples: int) -> list[list, list]:
    """5 fold cross validation

    Args:
        num_samples (int): total number of samples in the input data

    Returns:
        list[list, list]: A list where each entry contains a test set, and a training set
            in that order.
    """

    return k_fold(num_samples, 5)


def k_fold_10(num_samples: int) -> list[list, list]:
    """10 fold cross validation

    Args:
        num_samples (int): total number of samples in the input data

    Returns:
        list[list, list]: A list where each entry contains a test set, and a training set
            in that order.
    """

    return k_fold(num_samples, 10)


def leave_one_out(num_samples: int) -> list[list, list]:
    """leave_one_out cross validation

    Args:
        num_samples (int): total number of samples in the input data

    Returns:
        list[list, list]: A list where each entry contains a test set, and a training set
            in that order.
    """

    return k_fold(num_samples, num_samples)


def get(cross_validator_name: str) -> Callable:
    """_summary_

    Args:
        cross_validator_name (str): the name of the cross validator function
            to retrieve.

    Returns:
        Callable: cross validation function.
    """

    if cross_validator_name == "k_fold_5":
        return k_fold_5
    if cross_validator_name == "k_fold_10":
        return k_fold_10
    if cross_validator_name == "leave_one_out":
        return leave_one_out
