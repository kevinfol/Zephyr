import numpy as np
import copy
from src.data_parsing import InputData
from sklearn.model_selection import ParameterGrid
from src import scoring


def process_chromosome(
    chromosome: int,
    pipeline_obj: dict,
    regressor: object,
    preprocessor: object,
    cross_validator: callable,
    scorer: list[callable],
    input_data: InputData,
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
):

    # Make copies for the subprocess of all the relevant mutable objects
    regressor = copy.deepcopy(regressor)
    preprocessor = copy.deepcopy(preprocessor)
    cross_validator = copy.deepcopy(cross_validator)

    # Create the input feature data
    concat_list = []
    name_list = []
    for i, name in enumerate(input_data.feature_names):
        if chromosome & 2**i:
            name_list.append(name)
            concat_list.append(X[name])
    X_chromosome = np.column_stack(concat_list)

    # Create the target vector (mask out any missing feature data)
    mask = [~np.any(np.isnan(feature_row)) for feature_row in X_chromosome]
    X_chromosome = X_chromosome[mask].astype(np.float32)
    y_chromosome = y[mask].astype(np.float32)
    years_chromosome = years[mask]

    # retrieve the parameter grid for cv_testing
    param_grid = regressor.PARAM_GRID

    # Check if we need to optimize the no. of principal components
    if pipeline_obj.get("preprocessing", "standard") == "principal_components":
        possible_no_pcs = list(range(1, max(2, X_chromosome.shape[1])))
    else:
        possible_no_pcs = [None]

    param_grid["possible_no_pcs"] = possible_no_pcs
    param_grid = ParameterGrid(param_grid)

    best_params = param_grid[0]
    best_scores = [-np.inf for _ in scorer]

    for params in param_grid:

        # initialize the cv_predictions
        cv_predictions = np.full_like(y_chromosome, np.nan)
        cv_observations = np.full_like(y_chromosome, np.nan)

        # set up any monotone constraints
        monotonic = pipeline_obj.get("monotonic", False)
        if isinstance(monotonic, bool):
            monotone_constraints = [monotonic] * X_chromosome.shape[1]
        else:
            monotone_constraints = [
                input_data.feature_names.index(name) in monotonic for name in name_list
            ]

        if pipeline_obj.get("preprocessing", "standard") == "principal_components":
            # the number of PC's could be less than the number of predictors
            # NOTE: !! monotone constraints don't really work well for PC's .
            # you have to make the assumption that the first PC is "wetness", and
            # then make sure that one is monotonic, but maybe not the other PCs?
            # Either way, if you trace it all the way back to the original predictor
            # variables, there's no way to ensure that an increase in the original
            # predictor variable corresponds to an increase in the PC or the prediction.
            # For MLR, Ridge, and other coef/intercept methods, I guess you could compute
            # the actual coefficients for the original input predictors, and make sure they
            # are positive, but that would be cumbersome and not really in the spirit of
            # principal components. ... rant over ...

            monotone_constraints = [any(monotone_constraints)] + [False] * (
                params["possible_no_pcs"] - 1
            )

        regressor.set_monotonic(monotone_constraints)

        # fit the model with the selected params
        if "" not in params.keys():
            regressor.set_params(**params)

        # Fit the whole mode to check for non-importance predictors
        preprocessor.fit(X_chromosome)
        X_preprocessed = preprocessor.transform(X_chromosome)
        regressor.fit(X_preprocessed, y_chromosome)
        if regressor.has_zero_importance_predictors():
            scores = list(map(lambda s: -np.inf, scorer))

        else:
            # Iterate thru the CV and generate the cv_predictions for scoring
            for test_set, train_set in cross_validator(X_chromosome.shape[0]):
                X_train, y_train = X_chromosome[train_set], y_chromosome[train_set]
                X_test, y_test = X_chromosome[test_set], y_chromosome[test_set]

                # Preprocess the X data
                preprocessor.num_components = params["possible_no_pcs"]
                preprocessor.fit(X_train)
                X_train_preprocessed, X_test_preprocessed = (
                    preprocessor.transform(X_train),
                    preprocessor.transform(X_test),
                )

                # Fit the model
                regressor.fit(X_train_preprocessed, y_train, train_mode=True)

                # Make the cv_predictions
                cv_observations[test_set] = y_test
                cv_predictions[test_set] = regressor.predict(X_test_preprocessed)

            # Evaluate the score
            scores = list(
                map(
                    lambda s: s(
                        predicted=cv_predictions,
                        observed=cv_observations,
                        n_feats=X_chromosome.shape[1],
                    ),
                    scorer,
                )
            )

        if scoring.score_compare(scores, best_scores):
            best_scores = scores
            best_params = params
            cv_results = {
                "cv_predictions": cv_predictions,
                "observations": cv_observations,
            }

    # save the best score to the batch scores
    # batch_scores.append(best_scores)

    # save the best parameters for lookup later.
    # hyperparameters_table[chromosome] = best_params
    return chromosome, best_scores, best_params, cv_results
