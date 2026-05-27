# --------------------------------------------------------------------
# Config for test_statistics.py
# ---------------------------------------------------------------------

JSON_FILE_PATH = "testing/models_to_test/results/jsons/deep_ensemble_6_validation.json"
DIR_TO_SAVE = "testing/models_to_test/results/graphs"

SHOW_GRAPH = False
CONFIDENCE = 0.95
ACCEPTABLE_ERROR_THRESHOLD_MM = 25.0


import os
os.makedirs(DIR_TO_SAVE, exist_ok=True)

# --------------------------------------------------------------------
# Config for test_model.py
# ---------------------------------------------------------------------
PATH_TO_TEST_DATA_DIR = "data/regression_use/validation"
PATH_TO_MODEL = "testing/models_to_test/deep_ensemble_6.pth"
PATH_TO_RESULTS_FILE = JSON_FILE_PATH

INCLUDE_DEPTH = False
OUTPUT_BOUNDS = {
    "x_mean": 231.21374610473245,
    "x_std": 126.63452141714937,
    "y_mean": 206.52943690961092,
    "y_std": 94.59512251991019
}
BATCH_SIZE = 32


# Code so the filepaths exist
os.makedirs(os.path.dirname(PATH_TO_RESULTS_FILE), exist_ok=True)