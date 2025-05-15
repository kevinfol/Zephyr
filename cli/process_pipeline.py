import argparse
import os
import shutil
import sys

sys.path.append(os.getcwd())

import warnings

warnings.filterwarnings("ignore")
from functools import partial
import concurrent.futures
import json
import numpy as np
from uuid import uuid4
from multipledispatch import dispatch
from src import (
    data_parsing,
    regression,
    preprocessing,
    feature_selection,
    scoring,
    cross_validation,
    uncertainty_estimation,
    model_clustering,
    model_evaluation,
    misc_utils,
)
from random import randint, sample
from onnx.compose import merge_graphs
from onnx.helper import make_model, make_opsetid


@dispatch(dict, str, output_dir_path=str, clear_output=bool)
def process_pipeline(
    pipeline_obj: dict,
    input_data_path: str,
    output_dir_path: str = "~/ZephyrOutput",
    clear_output: bool = False,
) -> None:
    """Overload. Accepts path to input data file instead of InputData object.

    Args:
        pipeline_obj (dict): dictionary that fully defines a training pipeline.
        input_data (str): path to input data file containing the training data.
        output_dir_path (str, optional): directory to store the output results from
            the training. Defaults to "~/ZephyrOutput".
        clear_output (bool, optional): Set to `True` to delete all files in the
            output folder before training.
    """
    # expand the input path if needed
    input_data_path = os.path.expanduser(input_data_path)

    input_data = data_parsing.InputData(open(input_data_path, "r"))

    return process_pipeline(
        pipeline_obj,
        input_data,
        output_dir_path=output_dir_path,
        clear_output=clear_output,
    )


@dispatch(str, data_parsing.InputData, output_dir_path=str, clear_output=bool)
def process_pipeline(
    pipeline_json: str,
    input_data: data_parsing.InputData,
    output_dir_path: str = "~/ZephyrOutput",
    clear_output: bool = False,
) -> None:
    """Overload. Accepts a JSON string (either a path to a file, or a string of
        raw json) as the pipeline instead of a dict.

    Args:
        pipeline_json (str): either a str (of raw json) or a path to a json file
            where the pipeline json resides.
        input_data (InputData): InputData object containing the training data.
        output_dir_path (str, optional): directory to store the output results from
            the training. Defaults to "~/ZephyrOutput".
        clear_output (bool, optional): Set to `True` to delete all files in the
            output folder before training.
    """

    # Check if it's a valid file path
    if os.path.exists(os.path.expanduser(pipeline_json)):
        pipeline = json.load(open(os.path.expanduser(pipeline_json), "r"))
        return process_pipeline(
            pipeline,
            input_data,
            output_dir_path=output_dir_path,
            clear_output=clear_output,
        )

    # else, assume that it's a json raw string
    pipeline = json.loads(pipeline_json)
    return process_pipeline(
        pipeline, input_data, output_dir_path=output_dir_path, clear_output=clear_output
    )


@dispatch(str, str, output_dir_path=str, clear_output=bool)
def process_pipeline(
    pipeline_json: str,
    input_data_path: str,
    output_dir_path: str = "~/ZephyrOutput",
    clear_output: bool = False,
) -> None:
    """Overload. Accepts a JSON string (either a path to a file, or a string of
        raw json) as the pipeline instead of a dict. Also, accepts a file path to
        the input data.

    Args:
        pipeline_json (str): either a str (of raw json) or a path to a json file
            where the pipeline json resides.
        input_data (str): path to input data file containing the training data.
        output_dir_path (str, optional): directory to store the output results from
            the training. Defaults to "~/ZephyrOutput".
        clear_output (bool, optional): Set to `True` to delete all files in the
            output folder before training.
    """
    # expand the input path if needed
    input_data_path = os.path.expanduser(input_data_path)

    # process the input data
    input_data = data_parsing.InputData(open(input_data_path, "r"))
    return process_pipeline(
        pipeline_json,
        input_data,
        output_dir_path=output_dir_path,
        clear_output=clear_output,
    )


@dispatch(dict, data_parsing.InputData, output_dir_path=str, clear_output=bool)
def process_pipeline(
    pipeline_obj: dict,
    input_data: data_parsing.InputData,
    output_dir_path: str = "~/ZephyrOutput",
    clear_output: bool = False,
) -> None:
    """Processes a given training pipeline using the provided input data and
    outputting the results to the provided output directory. If the provided
    output directory doesn't exist, this script will try to create it.

    Example:
    >>> pipeline = {
        "regression_algorithm": "multiple_linear",
        "monotonic": [True, False, True, False, False]
        "cross_validation": "k_fold_5",
        "feature_selection": "genetic",
        "forced_features": [1,3],
        "preprocessing": "principal_components",
        "scoring": "adj_rsqrd",
        "exclude_years": [],
        "num_output_models": 3,
        }
    >>> input_data_obj = InputData(open('test_data.txt', 'r'))
    >>> output_dir = '~/Documents/ModelOutput'
    >>> process_pipeline(pipeline, input_data_obj, output_dir)

    Args:
        pipeline_obj (dict): dictionary that fully defines a training pipeline.
        input_data (InputData): InputData object containing the training data.
        output_dir_path (str, optional): directory to store the output results from
            the training. Defaults to "~/ZephyrOutput".
        clear_output (bool, optional): Set to `True` to delete all files in the
            output folder before training.
    """

    # Print the requested pipeline
    print("\n--------------------")
    print("Working on pipeline:")
    print("--------------------\n")
    print(json.dumps(pipeline_obj, indent=2) + "\n")

    print()

    # Expand the output path if needed
    output_dir_path = os.path.expanduser(output_dir_path.strip())

    # If the output directort doesn't exist, try to create it
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)
    else:
        # if we need to clear it out, do that
        if clear_output:
            for filename in os.listdir(output_dir_path):
                file_path = os.path.join(output_dir_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    # Initialize the JSON data
    json_output_data = {}

    # add the pipeline to the json output
    json_output_data["pipeline"] = pipeline_obj

    # Add the input data to the json output
    json_output_data["input_data"] = json.loads(input_data.make_json_metadata())

    # Save the input data layer of the model
    top_graph = input_data.make_onnx_graph()
    top_model = make_model(
        top_graph,
        opset_imports=[make_opsetid("", 19), make_opsetid("ai.onnx.ml", 2)],
    )
    with open(output_dir_path + "/input_data.onnx", "wb") as onnx_write_file:
        onnx_write_file.write(top_model.SerializeToString())

    # Initialize the regression object. By default: non-monotonic multiple linear
    regressor = regression.get(
        pipeline_obj.get("regression_algorithm", "multiple_linear")
    )()

    # Initialize the preprocessor. By default: standardization
    preprocessor = preprocessing.get(pipeline_obj.get("preprocessing", "standard"))(
        num_components=pipeline_obj.get("pc_num_components", 0)
    )

    # Initialize the feature selection algorithm. By default: Brute force with no features forced.
    feature_selector = feature_selection.get(
        pipeline_obj.get("feature_selection", "brute_force")
    )(
        num_features=len(input_data.feature_names),
        forced_features=pipeline_obj.get("forced_features", []),
        stopping_time=pipeline_obj.get("stopping_time", 600),
        selection_method=pipeline_obj.get("selection_method", "roulette"),
        population=pipeline_obj.get("genetic_pop_size", 11),
        generations=pipeline_obj.get("genetic_generations", 12),
        mutation_rate=pipeline_obj.get("genetic_mutation", 0.09),
    )

    # Initialize cross validator function
    cross_validator = cross_validation.get(
        pipeline_obj.get("cross_validation", "k_fold_5")
    )

    # Initialize scorer function. By default, adjusted r-squared is used.
    scorer = scoring.get(pipeline_obj.get("scoring", "adj_rsqrd"))

    # Filter out the exclude years from the input data
    mask = [
        year not in pipeline_obj.get("exclude_years", []) for year in input_data.years
    ]
    names = input_data.feature_names
    X = input_data.feature_data[mask]
    y = input_data.target_data[mask]
    years = input_data.years[mask]

    # Filter out any missing target data years from the input data
    mask = [~np.isnan(target_value) for target_value in y]
    X = X[mask]
    y = y[mask]
    years = years[mask]

    # Keep track of best hyperparameters per chromosome so we don't have
    # to re-figure-that-out once we determine the best chromosomes.
    hyperparameters_table = {}

    # Because computers nowadays have a lot of memory generally, we will keep
    # track of all the cross validated predictions and observations for
    # each chromosome so that we don't need to re-run cross validation
    # when we re-train the best models later.
    cv_results_table = {}

    # Check whether we should pre-seed.
    if len(names) >= 10:

        # Do a simple feature-wise correlation to find the best N/2 correlated
        # features
        corrs = []

        for feat in names:
            x_ = X[feat]
            r = np.corrcoef(x_, y)[0, 1]
            corrs.append((feat, r, names.index(feat)))

        corrs = sorted(corrs, key=lambda r: r[1], reverse=True)[: len(names) // 2]

        # create 10 random initial seeds
        seeds = []
        while len(seeds) < 10:
            k = randint(1, len(corrs))
            s = sample(corrs, k)
            val = sum([2 ** r[2] for r in s])
            if val not in seeds:
                seeds.append(val)
        feature_selector.preseed(seeds)
        if feature_selector.is_seeded:
            print(f"Preseeded feature selector with {len(seeds)} seeds\n")

    # Iterate through feature selection
    iteration_tracker = 0
    while feature_selector.is_still_running():

        # Get the batch of chromosomes that we'll evaluate in this iteration
        chromosome_batch = feature_selector.next()

        # Initialize a set of scores
        batch_scores = []

        # print progress
        feature_selector.print_progress(
            skip_preprint=iteration_tracker,
        )

        # Iterate thru the chromosomes and train the models

        # Parallelize the whole thing for maybe a significant speedup!
        if regressor.USE_PARALLEL_PROCESSING:
            executor_ = concurrent.futures.ProcessPoolExecutor
        elif regressor.USE_THREADING:
            executor_ = concurrent.futures.ThreadPoolExecutor
        else:
            executor_ = misc_utils.DummyExecutor
        with executor_() as executor:
            proc_func = partial(
                model_evaluation.process_chromosome,
                pipeline_obj=pipeline_obj,
                regressor=regressor,
                preprocessor=preprocessor,
                cross_validator=cross_validator,
                scorer=scorer,
                input_data=input_data,
                X=X,
                y=y,
                years=y,
            )

            for chromosome, best_scores, best_params, cv_results in executor.map(
                proc_func, chromosome_batch
            ):
                batch_scores.append(best_scores)
                hyperparameters_table[chromosome] = best_params
                cv_results_table[chromosome] = cv_results

        # set the scores for the feature selection algo to use in it's next iteration.
        feature_selector.set_scores(batch_scores)

        # update iteration tracker
        iteration_tracker += 1

    # print progress
    feature_selector.print_progress()

    # Use clustering to find the output models
    returned_best_chromosomes = model_clustering.distance_analysis(
        model_and_score_table=feature_selector.chromosome_scoring_table,
        num_models_returned=pipeline_obj.get("num_output_models", 1),
        num_features=len(input_data.feature_names),
    )

    # Retrain the best models and store the output
    for n, chromosome in enumerate(returned_best_chromosomes):

        # retrieve the stored best hyperparameters for this model
        hyperparameters = hyperparameters_table[chromosome]

        # Create the X_data, and Y_data for this model
        concat_list = []
        name_list = []
        for i, name in enumerate(input_data.feature_names):
            if chromosome & 2**i:
                name_list.append(name)
                concat_list.append(X[name])
        X_chromosome = np.column_stack(concat_list)

        # Create the target vector (mask out any missing feature data)
        mask = [~np.any(np.isnan(feature_row)) for feature_row in X_chromosome]
        X_chromosome = X_chromosome[mask]
        y_chromosome = y[mask]
        years_chromosome = years[mask]

        # Preprocess the X data
        preprocessor.num_components = hyperparameters["possible_no_pcs"]
        preprocessor.fit(X_chromosome)
        X_preprocessed = preprocessor.transform(X_chromosome)

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
                hyperparameters["possible_no_pcs"] - 1
            )

        regressor.set_monotonic(monotone_constraints)

        # Train the model with the selected params
        if "" not in hyperparameters.keys():
            regressor.set_params(**hyperparameters)
        regressor.fit(
            X_preprocessed.astype(np.float32), y_chromosome.astype(np.float32)
        )

        # Compute the cv-scores
        cv_predictions = cv_results_table[chromosome]["cv_predictions"]
        cv_observations = cv_results_table[chromosome]["observations"]
        num_predictors = sum(map(int, bin(chromosome)[2:]))
        scores = {}
        for scorer_name in scoring.ALL_SCORERS:
            scorer = scoring.get(scorer_name)[0]
            score = scorer(
                predicted=cv_predictions,
                observed=cv_observations,
                n_feats=num_predictors,
            )
            scores[scorer_name] = np.float64(score)

        # Create the model graph (preprocessor graph + regressor graph + uncertainty_graph)
        preprocessor_graph = preprocessor.create_onnx_graph(input_names=name_list)
        regressor_graph = regressor.create_onnx_graph()
        uncertainty_graph = uncertainty_estimation.create_uncertainty_onnx_graph(
            predictions=cv_predictions,
            observations=cv_observations,
            deg_freedom=num_predictors,
        )
        model_graph = merge_graphs(
            preprocessor_graph,
            regressor_graph,
            io_map=[("preprocessed_data", "x")],
        )
        model_graph = merge_graphs(
            model_graph,
            uncertainty_graph,
            io_map=[("y", "forecast_input")],
            outputs=["y", "forecast_uncertainty", "forecast_uncertainty_boxcox_or_log"],
        )

        # Save the individual model ONNX to the output directory
        onnx_filename = output_dir_path + str(uuid4())[::2] + ".onnx"
        with open(onnx_filename, "wb") as onnx_write_file:
            onnx_model = make_model(
                model_graph,
                opset_imports=[make_opsetid("", 19), make_opsetid("ai.onnx.ml", 2)],
            )
            onnx_write_file.write(onnx_model.SerializeToString())

        # save the json data
        # Store model info to the output json
        json_output_data[f"model_{n}_info"] = {
            "hyperparameters": hyperparameters,
            "cv_predictions": cv_predictions.astype(np.float64).tolist(),
            "observations": cv_observations.astype(np.float64).tolist(),
            "years": years_chromosome.tolist(),
            "chromosome": chromosome,
            "input_features": name_list,
            "scores": scores,
            "onnx": onnx_filename,
        }
        print(json_output_data[f"model_{n}_info"]["hyperparameters"])
        print(json_output_data[f"model_{n}_info"]["scores"])
        print()
        with open(output_dir_path + "/output.json", "w") as output_json_file:
            output_json_file.write(json.dumps(json_output_data, indent=2))

        print(
            "Saved model with chromosome: ",
            feature_selector.convert_number_to_bool_str(chromosome),
        )
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Zephyr Pipeline Processing",
        description="processes the provided training pipeline.",
        epilog="c.2025, author: K.Foley",
    )
    parser.add_argument(
        "pipeline",
        help="Either a path to a JSON file containing the pipeline, or the raw JSON containing the pipeline.",
    )
    parser.add_argument("input_data", help="The path to the input_data file.")
    parser.add_argument(
        "--output_dir",
        "-o",
        help="The path to the directory where output data will be saved.",
        required=False,
        default="~/ZephyrOutput",
    )
    parser.add_argument(
        "--clear-output-dir",
        "-c",
        help="Optionally clear out the output folder before running the pipeline.",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    process_pipeline(
        args.pipeline,
        args.input_data,
        output_dir_path=args.output_dir,
        clear_output=args.clear_output_dir,
    )
