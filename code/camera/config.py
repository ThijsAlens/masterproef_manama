"""
This file contains configuration settings for the Camera class.
"""

WIDTH = 640               # Width of the camera frame
HEIGHT = 480              # Height of the camera frame
FPS = 30                  # Frames per second for the camera

CHECKERBOARD_SIZE = (6, 9)  # Number of inner corners per a chessboard row and column
SQUARE_SIZE_M = 0.0253      # Size of a square in your defined unit

DEFAULT_CROP_X = (0, 550)  # Default cropping range for the x-axis (horizontal)
DEFAULT_CROP_Y = (0, 320)  # Default cropping range for the y-axis (vertical)

EXCLUDE_MEASURING_X = (0, 480)  # Cropping range for the x-axis to exclude measuring lines
EXCLUDE_MEASURING_Y = (0, 480)  # Cropping range for the y-axis to exclude measuring lines

CALIBRATION_IMAGE_PATH = "camera/checkerboard.png"                         # Path to the calibration image file
CALIBRATION_REAL_WORLD_PATH = "camera/lines.png"
CALIBRATION_MAPPING_FILEPATH = "data/regression/calibration/pixel_to_real_world_mapping.json"  # Path to save/load the pixel-to-real-world mapping
CALIBRATION_T_MATRIX_FILEPATH = "data/regression/calibration/camera_translation_matrix.npy"     # Path to save/load the camera translation matrix
CALIBRATION_R_MATRIX_FILEPATH = "data/regression/calibration/camera_rotation_matrix.npy"        # Path to save/load the camera rotation matrix

CANNY_LOW_THRESHOLD = 40     # Lower threshold for Canny edge detection
CANNY_HIGH_THRESHOLD = 107   # Higher threshold for Canny edge detection
DISTANCE_BETWEEN_LINES_MM = 200 # Real-world distance between the lines in millimeters (used for scaling)