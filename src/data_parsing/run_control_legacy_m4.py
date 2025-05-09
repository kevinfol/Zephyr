from configparser import ConfigParser
from argparse import FileType
from typing import Any

ARG_MAP = {
    "basic": ["RunTypeFlag", "errorlog_flag"],
    "features": [
        "GeneticAlgorithmFlag",
        "MaxModes",
        "DisplayFlag",
        "MinNumVars",
        "GAPopSize",
        "GANumGens",
        "ManualPCSelection",
    ],
    "ensemble": [
        "AutoEnsembleFlag",
        "EnsembleFlag_PCR",
        "EnsembleFlag_PCR_BC",
        "EnsembleFlag_PCQR",
        "EnsembleFlag_PCANN",
        "EnsembleFlag_PCANN_BC",
        "EnsembleFlag_PCRF",
        "EnsembleFlag_PCRF_BC",
        "EnsembleFlag_PCMCQRNN",
        "EnsembleFlag_PCSVM",
        "EnsembleFlag_PCSVM_BC",
    ],
    "ann": [
        "AutoANNConfigFlag",
        "ANN_config1_cutoff",
        "mANN_config_selection",
        "MCQRNN_config_selection",
        "ANN_monotone_flag",
        "ANN_parallel_flag",
        "num_cores",
    ],
    "svm": ["SVM_config_selection", "fixedgamma"],
    "outputs": ["PC1form_plot_flag", "PC12form_plot_flag"],
    "forward_pca": [
        "PCSelection_Frwrd_LR",
        "PCSelection_Frwrd_QR",
        "PCSelection_Frwrd_mANN",
        "PCSelection_Frwrd_MCQRNN",
        "PCSelection_Frwrd_RF",
        "PCSelection_Frwrd_SVM",
    ],
    "forward_input_variables": [
        "VariableSelection_Frwrd_LR",
        "VariableSelection_Frwrd_QR",
        "VariableSelection_Frwrd_mANN",
        "VariableSelection_Frwrd_MCQRNN",
        "VariableSelection_Frwrd_RF",
        "VariableSelection_Frwrd_SVM",
    ],
    "forward_ann": ["ANN_monotone_flag_Frwrd"],
    "forward_ensemble": ["Ensemble_flag_frwrd", "Ensemble_type_frwrd"],
}


def parse_run_control_file_legacy(file_handle: FileType) -> ConfigParser:
    """Takes a run-control file that is opened in text mode (not binary)
    and creates a ConfigParser object containg all the selected options/settings.

    Example:
    >>> c = parse_run_control_file(rc_file)
    >>> print(c["feature_processing_and_selection"]["GAPopSize"])
    10

    Args:
        file_handle (FileType): a file-like object containing the Zephyr/M4 run
            control file. The file must be opened in text mode (not binary)

    Returns:
        ConfigParser: A ConfigParser object that contains all the settings/sections.
    """

    # Create the ConfigParser
    config_parser = ConfigParser()

    # Split the file into lines and remove empty lines
    rc_lines = file_handle.read().split("\n")
    rc_lines = filter(lambda line: line.strip() != "", rc_lines)

    # Create the dictionary of settings
    settings_dict = {k: {f: None for f in v} for k, v in ARG_MAP.items()}

    # Parse each line into it's correct setting
    for line in rc_lines:

        # Split the line into the value and the description
        value, description = line.split("#")
        key = description.strip().split(" ")[0]
        section = get_section(key)

        # If the section exists, set the value
        if section:
            value = parse_value(value)
            if isinstance(value, str):
                if value.isnumeric():
                    print()
            settings_dict[section][key] = value

    # Set the dict to the ConfigParser
    config_parser.read_dict(settings_dict)
    return config_parser


def get_section(key: str) -> str:
    """returns the configuration section associated with the argument

    Args:
        key (str): the argument to find the section name for
    """
    for section in ARG_MAP.keys():
        if key in ARG_MAP[section]:
            return section
    return None


def parse_value(value_string: str) -> Any:
    """Attempts to convert run control parameters into python
    types. Tries to handle booleans, integers, floats, and lists.

    Args:
        value_string (str): The run control parameter value that will be
            converted.

    Returns:
        Any: The python conversion of the run control parameter
    """

    # attempt to process lists (e.g. "Y","N" would process to [True, False])
    if "," in value_string:
        return [parse_value(value) for value in value_string.split(",")]

    # Attemp to process Bools
    if '"' in value_string:
        if value_string.strip() == '"Y"':
            return True
        if value_string.strip() == '"N"':
            return False
        return parse_value(value_string.replace('"', "").strip())

    # Process numbers
    if value_string.strip().strip(".").isnumeric():
        if "." in value_string:
            return float(value_string.strip())
        else:
            return int(value_string.strip())

    return value_string.strip()
