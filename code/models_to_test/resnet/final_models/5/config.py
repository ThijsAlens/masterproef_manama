# Val loss = -2.1212
SEED = 1

PATH_TO_TRAIN_DATA_DIR = "data/regression_use/train"
PATH_TO_VALIDATION_DATA_DIR = "data/regression_use/validation"
PATH_TO_TEST_DATA_DIR = "data/regression_use/test"

PATH_TO_SAVE_MODEL_DIR = "models_to_test/resnet/models"

PATH_TO_RESULTS_DIR = "models_to_test/resnet/results"

MODEL_PARAMETERS = {
    "input_dims": (3, 370, 250),
    "output_size": 2, # (x, y)
    "freeze_backbone": True
}
NUMBER_OF_RES_BLOCKS = 4                # (total number of blocks is 8)

OUTPUT_BOUNDS = {
    "x_mean": 231.21374610473245,
    "x_std": 126.63452141714937,
    "y_mean": 206.52943690961092,
    "y_std": 94.59512251991019
}

INCLUDE_DEPTH = False

NUM_MODELS = 5
EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 16
BAGGING_SAMPLE_RATIO = None