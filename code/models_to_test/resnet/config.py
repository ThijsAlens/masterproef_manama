SEED = 1

PATH_TO_TRAIN_DATA_DIR = "data/regression_use/train"
PATH_TO_VALIDATION_DATA_DIR = "data/regression_use/validation"
PATH_TO_TEST_DATA_DIR = "data/regression_use/test"

PATH_TO_SAVE_MODEL_DIR = "models_to_test/resnet/models"

PATH_TO_RESULTS_DIR = "models_to_test/resnet/results"

MODEL_PARAMETERS = {
    "input_dims": (3, 370, 250),
    "output_size": 2, # (x, y)
    "freeze_backbone": False
}

OUTPUT_BOUNDS = {
    "x_mean": 231.21374610473245,
    "x_std": 126.63452141714937,
    "y_mean": 206.52943690961092,
    "y_std": 94.59512251991019
}

INCLUDE_DEPTH = False                   # False

NUM_MODELS = 3                          # 3
EPOCHS = 100                            # 100
LR = 1e-5                               # 1e-5
BATCH_SIZE = 32                         # 32
BAGGING_SAMPLE_RATIO = 0.9              # 0.8