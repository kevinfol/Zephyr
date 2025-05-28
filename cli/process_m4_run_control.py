import argparse
import sys, os

sys.path.append(os.getcwd())
from src import data_parsing
from cli import process_pipeline


def process_m4(
    run_control_path: str,
    input_data_path: str,
    output_dir: str = "~/ZephyrOutput",
    clear_output_dir: bool = False,
) -> None:
    """Processes a M4-style run-control file + input data and outputs the
    resulting 6 models to the specified directory. For each ensemble member
    a pipeline is created with PC transformation and a genetic algorithm (if
    specified in the run-control (otherwise, all features are forced))

    Args:
        run_control_path (str): A path to a valid M4 run control file (e.g.
            "C:/Users/JohnDoe/Documents/WrongRiverRunControl.txt")
        input_data_path (str): A path to a valid M4 run datal file (e.g.
            "C:/Users/JohnDoe/Documents/WrongRiverTrainData.txt")
        output_dir (str, optional): A path to a directory where outputs will be
            stored. If the directory doesn't exist, it will be created.
            Defaults to "~/ZephyrOutput".
        clear_output_dir (bool, optional): Optionally clear the output directory
            if it exists and already has files in it. Defaults to False.
    """

    # Open the run control file and read it if possible
    with open(run_control_path, "r") as run_control_file:
        run_control = data_parsing.parse_run_control_file_legacy(run_control_file)

    # Open the input data file and convert it to InputData Class
    with open(input_data_path, "r") as input_data_file:
        input_data = data_parsing.InputData(input_data_file)

    # Determine if we're in GA mode or All-Features-Non-GA mode
    if run_control["features"].getboolean("GeneticAlgorithmFlag"):
        use_genetic_algorithm = True
        forced_features = []
        ga_pop_size = run_control["features"].getint("GAPopSize")
        ga_generations = run_control["features"].getint("GANumGens")
    else:
        use_genetic_algorithm = False
        forced_features = list(range(len(input_data.feature_names)))
        ga_pop_size = None
        ga_generations = None

    # Create the pipelines
    pipelines = [
        # PCR Regression (not monotonic by default)
        {
            "regression_algorithm": "multiple_linear",
            "monotonic": False,
            "feature_selection": "genetic" if use_genetic_algorithm else "brute_force",
            "genetic_pop_size": ga_pop_size,
            "genetic_generations": ga_generations,
            "forced_features": forced_features,
            "preprocessing": "principal_components",
            "scoring": "d2_rmse",
            "exclude_years": [],
            "num_output_models": 1,
            "pc_num_components": run_control["features"].getint("MaxModes"),
        },
        # Random Forest Regression (not monotonic by default)
        {
            "regression_algorithm": "random_forest",
            "monotonic": False,
            "feature_selection": "genetic" if use_genetic_algorithm else "brute_force",
            "genetic_pop_size": ga_pop_size,
            "genetic_generations": ga_generations,
            "forced_features": forced_features,
            "preprocessing": "principal_components",
            "scoring": "d2_rmse",
            "exclude_years": [],
            "num_output_models": 1,
            "pc_num_components": run_control["features"].getint("MaxModes"),
        },
        # Support Vector Machine (not monotonic by default)
        {
            "regression_algorithm": "support_vector",
            "monotonic": False,
            "feature_selection": "genetic" if use_genetic_algorithm else "brute_force",
            "genetic_pop_size": ga_pop_size,
            "genetic_generations": ga_generations,
            "forced_features": forced_features,
            "preprocessing": "principal_components",
            "scoring": "d2_rmse",
            "exclude_years": [],
            "num_output_models": 1,
            "pc_num_components": run_control["features"].getint("MaxModes"),
        },
        # Monotonic Neural Network (monotonic)
        {
            "regression_algorithm": "neural_network",
            "monotonic": True,
            "feature_selection": "genetic" if use_genetic_algorithm else "brute_force",
            "genetic_pop_size": ga_pop_size,
            "genetic_generations": ga_generations,
            "forced_features": forced_features,
            "preprocessing": "principal_components",
            "scoring": "d2_rmse",
            "exclude_years": [],
            "num_output_models": 1,
            "pc_num_components": run_control["features"].getint("MaxModes"),
        },
    ]

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Zephyr M4 Run Control File processing",
        description="Processes a legacy M4 run control file (along with an"
        "input data file.)",
        epilog="c.2025, author: K.Foley",
    )
    parser.add_argument(
        "run_control_file",
        help="The path to a valid M4 run control file.",
    )
    parser.add_argument(
        "input_data_file",
        help="The path to a valid M4 input_data file.",
    )
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
    process_m4(
        args.pipeline,
        args.input_data,
        output_dir_path=args.output_dir,
        clear_output=args.clear_output_dir,
    )
