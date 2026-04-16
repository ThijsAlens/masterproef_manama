ROOT_DIRECTORY = "data/regression" # Change this to "data/object-detection" for the object detection dataset
CLASSES = \
    {
        "disk": 0,
        "ball": 1,
    }

IMAGE_ID_START = 0 # Change this if you already have images in the dataset folder to avoid overwriting

NUMBER_OF_INNER_CORNERS_X = 9 # Number of inner corners in the chessboard pattern along the x-axis (width)
NUMBER_OF_INNER_CORNERS_Y = 6 # Number of inner corners in the chessboard pattern along the y-axis (height)
SQUARE_SIZE_MM = 23 # The size of one square in the chessboard pattern in millimeters

CANNY_LOW_THRESHOLD = 40     # Lower threshold for Canny edge detection (used for auto-detection of disk centers)
CANNY_HIGH_THRESHOLD = 120   # Higher threshold for Canny edge detection (used for auto-detection of disk centers)