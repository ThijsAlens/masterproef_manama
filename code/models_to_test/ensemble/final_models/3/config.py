# Val loss = -2.6241

SEED = 1

PATH_TO_TRAIN_DATA_DIR = "data/regression_use/train"
PATH_TO_VALIDATION_DATA_DIR = "data/regression_use/validation"
PATH_TO_TEST_DATA_DIR = "data/regression_use/test"

PATH_TO_SAVE_MODEL_DIR = "models_to_test/ensemble/models"

PATH_TO_RESULTS_DIR = "models_to_test/ensemble/results"

MODEL_PARAMETERS = {
    "input_dims": (3, 370, 250),
    "hidden_channels": [16, 32, 64],
    "output_size": 2 # (x, y)
}

OUTPUT_BOUNDS = {
    "x_mean": 231.21374610473245,
    "x_std": 126.63452141714937,
    "y_mean": 206.52943690961092,
    "y_std": 94.59512251991019
}

INCLUDE_DEPTH = False

NUM_MODELS = 5
EPOCHS = 60
LR = 5e-4
BATCH_SIZE = 32
BAGGING_SAMPLE_RATIO = None