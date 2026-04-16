SEED = 1

PATH_TO_TRAIN_DATA_DIR = "data/regression_use/train"
PATH_TO_VALIDATION_DATA_DIR = "data/regression_use/validation"
PATH_TO_TEST_DATA_DIR = "data/regression_use/test"

PATH_TO_SAVE_MODEL_DIR = "code/models_to_test/ensemble/models"

MODEL_PARAMETERS = {
    "input_channels": 3,
    "hidden_channels": [64, 32, 8],     # [64, 32, 8]
    "output_size": 2 # (x, y)
}

INPUT_DIMENSION = (3, 370, 250)
INCLUDE_DEPTH = False                   # False

NUM_MODELS = 3                          # 3
EPOCHS = 100                            # 100
LR = 1e-3                               # 1e-3
BATCH_SIZE = 32                         # 32
BAGGING_SAMPLE_RATIO = 0.8              # 0.8