# --------------------------------------------------------------------
# Config for test_statistics.py
# ---------------------------------------------------------------------

JSON_FILE_PATH = "models_to_test/MC_dropout/results/test_results_2026-05-13.json"
DIR_TO_SAVE = "testing/graphs"

SHOW_GRAPH = True
CONFIDENCE_INTERVAL = 0.95
ACCEPTABLE_ERROR_THRESHOLD_MM = 20.0


import os
os.makedirs(DIR_TO_SAVE, exist_ok=True)

# --------------------------------------------------------------------
# Config for test_model.py
# ---------------------------------------------------------------------
PATH_TO_TEST_DATA_DIR = "data/test/human"
PATH_TO_MODEL = ""
PATH_TO_RESULTS_FILE = "testing/model_tests/ensemble.json"

INCLUDE_DEPTH = False
OUTPUT_BOUNDS = {
    "x_mean": 231.21374610473245,
    "x_std": 126.63452141714937,
    "y_mean": 206.52943690961092,
    "y_std": 94.59512251991019
}
BATCH_SIZE = 32


os.makedirs(os.path.dirname(PATH_TO_RESULTS_FILE), exist_ok=True)