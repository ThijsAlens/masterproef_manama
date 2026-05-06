SEED = 1

PATH_TO_TRAIN_DATA_DIR = "data/regression_use/train"
PATH_TO_VALIDATION_DATA_DIR = "data/regression_use/validation"
PATH_TO_TEST_DATA_DIR = "data/regression_use/test"

PATH_TO_SAVE_MODEL_DIR = "models_to_test/MC_dropout/models"

PATH_TO_RESULTS_DIR = "models_to_test/MC_dropout/results"

MODEL_PARAMETERS = {
    "input_channels": 3,
    "hidden_channels": [32, 64, 128],     # [32, 64, 128]
    "output_size": 2,
    "p_list": [0.05, 0.05, 0.05]           # [0.5, 0.5, 0.5]
}

OUTPUT_BOUNDS = {
    "x_mean": 231.21374610473245,
    "x_std": 126.63452141714937,
    "y_mean": 206.52943690961092,
    "y_std": 94.59512251991019
}

INPUT_DIMENSION = (3, 370, 250)
INCLUDE_DEPTH = False                   # False

NUM_SAMPLES = 100
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 32
BAGGING_SAMPLE_RATIO = 0.8