SEED = 1

PATH_TO_TRAIN_DATA_DIR = "path/to/training_dataset"
PATH_TO_VALIDATION_DATA_DIR = "path/to/validation_dataset"
PATH_TO_TEST_DATA_DIR = "path/to/test_dataset"

MODEL_PARAMETERS = {
    "input_channels": 3,
    "hidden_channels": [16, 32],
    "output_size": 2
}
INPUT_DIMENSION = (3, 64, 64)

NUM_MODELS = 5
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
BAGGING_SAMPLE_RATIO = 0.8