import sys, os

sys.path.append(os.getcwd())

import unittest
from data_parsing import parse_run_control_file_legacy
from io import StringIO

DATA_FILE_IO = StringIO(
    """# RUN CONTROL (Set leftmost column to desired values; use all caps for character variables; keep the quotation marks where they are for those variables that use them and do not use them otherwise)



# BASIC PARAMETERS: CONTROL WHETHER INVERSE OR FORWARD RUN, AND WHETHER MESSAGES ARE SENT TO AN EXTERNAL LOG FILE


"BUILD"       # RunTypeFlag               - If set to "BUILD" then the model development process is undertaken; if set to "FORECAST" then previously built ensemble of models are run with new sample of predictor data
"Y"           # errorlog_flag             - Log error & warning messages to external text file?  This will suppress some on-screen output



# RUN PARAMETERS FOR MODEL DEVELOPMENT (Used only if RunTypeFlag = "BUILD")


# CONTROL FEATURE CREATION & SELECTION:

"Y"           # GeneticAlgorithmFlag      - Use genetic algorithm for optimal predictor selection?  If "Y", code finds optimal combination of input variables, and which PCA modes to retain, separately for each of the modeling technologies, and in the process fits each of the models
2             # MaxModes                  - If GeneticAlgorithmFlag = "Y" (ignored otherwise): specify maximum allowable number of PCA modes to retain, counting upwawrd from the leading mode (modes may be skipped; 4th PCA mode is highest allowable)
"Y"           # DisplayFlag               - If GeneticAlgorithmFlag = "Y" (ignored otherwise): display to screen progress of genetic algorithm? (note: progress written to file irrespective of this setting)
2             # MinNumVars                - If GeneticAlgorithmFlag = "Y" (ignored otherwise): minimum allowable number of input variables to retain
12            # GAPopSize                 - If GeneticAlgorithmFlag = "Y" (ignored otherwise): population size
10             # GANumGens                 - If GeneticAlgorithmFlag = "Y" (ignored otherwise): number of generations
1,2           # ManualPCSelection         - If GeneticAlgorithmFlag = "N" (ignored otherwise): specify which PCA modes to fit the models to (1=leading mode, 2 = second mode, etc.; use comma between modes if more than one retained; same modes will be used for all modeling techniques; all input variables in the input data file will be used; modes may be skipped; 4th PCA mode is highest allowable)

# CONTROL ENSEMBLE GENERATION:

"Y"           # AutoEnsembleFlag          - Automatic ensemble generation?  If "Y", code starts with default ensemble members, automatically checks ensemble output for non-physical (negative) values, and adjusts member composition accordingly
"Y"           # EnsembleFlag_PCR          - If AutoEnsembleFlag = "N" (ignored otherwise): include output from conventional PCR in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"N"           # EnsembleFlag_PCR_BC       - If AutoEnsembleFlag = "N" (ignored otherwise): include output from PCR with Box-Cox transform-space prediction bounds in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"Y"           # EnsembleFlag_PCQR         - If AutoEnsembleFlag = "N" (ignored otherwise): include output from PCQR in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"Y"           # EnsembleFlag_PCANN        - If AutoEnsembleFlag = "N" (ignored otherwise): include output from ANN with conventional prediction bounds in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"N"           # EnsembleFlag_PCANN_BC     - If AutoEnsembleFlag = "N" (ignored otherwise): include output from ANN with Box-Cox prediction bounds in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"Y"           # EnsembleFlag_PCRF         - If AutoEnsembleFlag = "N" (ignored otherwise): include output from PCRF in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"N"           # EnsembleFlag_PCRF_BC      - If AutoEnsembleFlag = "N" (ignored otherwise): include output from PCRF in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"Y"           # EnsembleFlag_PCMCQRNN     - If AutoEnsembleFlag = "N" (ignored otherwise): include output from PCRF in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"Y"           # EnsembleFlag_PCSVM        - If AutoEnsembleFlag = "N" (ignored otherwise): include output from PCSVM in multi-model ensemble?  Capital "Y" = yes, capital "N" = no
"N"           # EnsembleFlag_PCSVM_BC     - If AutoEnsembleFlag = "N" (ignored otherwise): include output from PCSVM in multi-model ensemble?  Capital "Y" = yes, capital "N" = no

# CONTROL CONFIGURATION & PARALLELIZATION OF ANNS (BOTH mANN AND MCQRNN):

"Y"           # AutoANNConfigFlag         - Automatic ANN configuration selection generation?  If "Y", code starts with basic configuration, automatically compares ANN prediction quality to that of other models, and uses alternative configuration if necessary and appropriate
25            # ANN_config1_cutoff        - If AutoANNConfigFlag = "Y" (ignored otherwise): max allowable % performance deficit in R^2 or RMSE of default config relative to other models
1             # mANN_config_selection     - If AutoANNConfigFlag = "N" (ignored otherwise): if 1, use basic mANN configuration (1 hidden-layer neuron with no bagging), if 2 use standard backup (alternative) configuration (2 hidden-layer neurons with 20 bootsraps), if 3 use custom configuration (hard-wired by experienced user within ANN module)
1             # MCQRNN_config_selection   - If AutoANNConfigFlag = "N" (ignored otherwise): if 1, use basic ANN configuration (1 hidden-layer neuron with no bagging), if 2 use standard backup (alternative) configuration (2 hidden-layer neurons with 20 bootsraps), if 3 use custom configuration (hard-wired by experienced user within ANN module)
"Y"           # ANN_monotone_flag         - Enable monotonicity constraint?  If "Y" then relationships between all predictors and predictand will be forced to be monotonic; not used if configuration selection is 3
"Y"           # ANN_parallel_flag         - Enable parallelization?  If "Y" then foreach loop is used to perform cross-validation
8             # num_cores                 - If parallel_flag = "Y" (ignored otherwise): set to number of processor cores across which task will be distributed

# CONTROL SVM CONFIGURATION:

2             # SVM_config_selection      - Select options for automated SVM hyperparameter tuning: if 1, tune gamma, epsilon, and cost; if 2 tune epsilon and cost only; if 3 no automated tuning is performed and hyperparameters are hard-wired by experienced user at corresponding location within SVM module
0.2112        # fixedgamma                - If SVM_config_selection = 2 (ignored otherwise): fixed value of gamma to use

# CONTROL SOME OUTPUTS:

"Y"           # PC1form_plot_flag         - Create PC1 vs. obs scatter plots for each model (irrespective of whether additional PCs are selected as predictive variates)?
"Y"           # PC12form_plot_flag        - Create PC1-PC2-obs contour plots for each model (only performed if PC2 is available)?



# RUN PARAMETERS FOR FORECASTING (Used only if RunTypeFlag = "FORECAST")


# SPECIFY WHICH PCA MODES TO RETAIN: 

1           # PCSelection_Frwrd_LR       - For PCR (1=leading mode, 2-second mode, etc.; a combination of up to the first four PCs, determined by what was manually specified or determined by genetic algorithm during model-building for PCR; if more than one mode, separate by commas)
1           # PCSelection_Frwrd_QR       - For PCQR 
1           # PCSelection_Frwrd_mANN     - For PCANN
1           # PCSelection_Frwrd_MCQRNN   - For PCMCQRNN
1           # PCSelection_Frwrd_RF       - For PCRF
1           # PCSelection_Frwrd_SVM      - For PCSVM

# SPECIFY WHICH INPUT VARIABLES TO RETAIN: 1 TO KEEP, 0 TO OMIT, POSITIONS CORRESPOND TO LEFT-TO-RIGHT CONSECUTIVE LOCATIONS IN INPUT DATASET

"1"       # VariableSelection_Frwrd_LR       - For PCR (set to all the input variables used to build the model if genetic algorithm was not used, or the optimal combination determined by genetic algorithm during model building; separate by spaces as in genetic algorithm run summary file)
"1"      # VariableSelection_Frwrd_QR       - For PCQR
"1"     # VariableSelection_Frwrd_mANN     - For PCANN
"1"  # VariableSelection_Frwrd_MCQRNN   - For MCQRNN
"1"      # VariableSelection_Frwrd_RF       - For PCRF
"1"     # VariableSelection_Frwrd_SVM      - For PCSVM

# HOUSEKEEPING FOR MONOTONIC NEURAL NETS:

"Y"           # ANN_monotone_flag_Frwrd    - Was the monotonicity constraint enabled for the neural networks (PCANN, PCMCQRNN) in the inverse run?

# SPECIFY WHETHER ENSEMBLE MEAN FORECAST DISTRIBUTION IS TO BE GENERATED AND IF SO FROM WHICH MODELS

"Y"           # Ensemble_flag_frwrd        - Calculate the average of the forecast distributions from the individual models in the forward runs?  (If "N" then only forecast distributions from individual models are written to file; if "Y" then ensemble mean is additionally calculated and written to file)
"AUTO"        # Ensemble_type_frwrd        - If Ensemble_flag_frwrd = "Y" (ignored otherwise): "ALL" creates ensemble mean from default model set (PCR-BC, PCQR, PCANN-BC, PCMCQRNN, PCRF-BC, PCSVM-BC), "AUTO" does the same but potentially removes individual ensemble members in accordance with AutomatedEnsembleMemberExclusions.csv

"""
)
DATA_FILE_IO.name = "testfile.txt"


class TestM4RunControl(unittest.TestCase):

    def test_1(self):

        print("Data Parsing: Testing legacy M4 run control")
        DATA_FILE_IO.seek(0)
        rc_config = parse_run_control_file_legacy(DATA_FILE_IO)
        self.assertEqual(rc_config["features"]["MaxModes"], "2")
        print()


if __name__ == "__main__":
    unittest.main()
