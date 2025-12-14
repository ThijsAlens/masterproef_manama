"""
This file contains configuration settings for the Camera class.
"""

WIDTH = 640               # Width of the camera frame
HEIGHT = 480              # Height of the camera frame
FPS = 30                  # Frames per second for the camera

CHECKERBOARD_SIZE = (6, 9)  # Number of inner corners per a chessboard row and column
SQUARE_SIZE_M = 0.0253      # Size of a square in your defined unit

CALIBRATION_IMAGE_PATH = "calibration/checkerboard.png"                         # Path to the calibration image file
CALIBRATION_T_MATRIX_FILEPATH = "calibration/camera_translation_matrix.npy"     # Path to save/load the camera translation matrix
CALIBRATION_R_MATRIX_FILEPATH = "calibration/camera_rotation_matrix.npy"        # Path to save/load the camera rotation matrix