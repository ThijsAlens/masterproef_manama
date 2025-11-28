import datetime
import pyrealsense2 as rs
import numpy as np
import cv2

import camera.config as cam_config

class RealSenseCamera:
    """
    This class is used for everything to do with the RealSense D415 camera.
    It handles the setup, automatically calibrates the camera to a checkerboard pattern.
    """

    # --- 2. Constructor and Initialization ---
    def __init__(self):
        # Camera settings
        self.width = cam_config.WIDTH
        self.height = cam_config.HEIGHT
        self.fps = cam_config.FPS
        
        # Calibration matrices
        self.K_matrix = None            # Camera Matrix (K), this is the intrinsic matrix which corrects for focal length and principal point
        self.D_matrix = None            # Distortion Coefficients (D), for lens distortion correction
        self.rs_intrinsics = None       # RealSense Intrinsics object, used for the translation from pixel to camera coordinates
        self.R_matrix = None            # Rotation Matrix (World to Camera)
        self.T_vector = None            # Translation Vector (World to Camera)

        # RealSense pipeline setup
        self.pipeline = rs.pipeline()   # Create a RealSense pipeline, this is used to configure, start and stop the camera
        self.config = rs.config()       # Create a config object to configure the pipeline
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)   # Enable depth stream
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # Enable color stream
        return

    def start_stream(self):
        """Starts the RealSense pipeline and get the intrinsics.

        Args:
            None
        
        Returns:
            profile: The RealSense pipeline profile.
        """
        profile = self.pipeline.start(self.config)

        # Get the profile for the depth stream and extract factory intrinsics
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.rs_intrinsics = depth_profile.get_intrinsics()
        return profile
    
    def stop_stream(self):
        """Stops the RealSense pipeline."""
        self.pipeline.stop()
        print("RealSense stream stopped.")
    
    def setup_matrices(self, mode: str = "live", file_path_T: str = None, file_path_R: str = None, file_path_image: str = None) -> None:
        """
        Sets up the intrinsic (K, D) and extrinsic (R, T) matrices for the camera.
        Intrinsic matrices are calculated using the RealSense intrinsics, if they are not set, run start_stream() first.
        Extrinsic matrices (R and T) are:
            - loaded from a previous run (specify file paths) (mode="load")
            - calculated from live calibration (needs the pipeline, if not active, run start_stream()) (mode="live", default)
            - calculated from a checkerboard image (setup_live=False, specify file path) (mode="image")

        Args:
            mode (str): "live", "load", or "image" to specify how to set up R and T. (default: "live")
            file_path_T (str): File path to load the T vector from (required if mode="load").
            file_path_R (str): File path to load the R matrix from (required if mode="load").
            file_path_image (str): File path to load the checkerboard image from (required if mode="image").

        Returns:
            None
        """

        if self.rs_intrinsics is None:
            raise ValueError("RealSense intrinsics not set. Call start_stream() first.")
        if mode != "live" and mode != "load" and mode != "image":
            raise ValueError("Invalid mode. Choose 'live', 'load', or 'image'.")
        
        # --- Intrinsic Matrices Setup ---
        
        self.K_matrix = np.array([
            [self.rs_intrinsics.fx, 0, self.rs_intrinsics.ppx],
            [0, self.rs_intrinsics.fy, self.rs_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)

        self.D_matrix = np.array(self.rs_intrinsics.coeffs, dtype=np.float64)

        # --- Extrinsic Matrices Setup ---

        match mode:
            case "load":
                if file_path_T is None or file_path_R is None:
                    raise ValueError("File paths for T and R matrices must be provided to load previous setup.")
                self.T_vector = np.loadtxt(file_path_T)
                self.R_matrix = np.loadtxt(file_path_R)
                return

            case "image":
                if file_path_image is None:
                    raise ValueError("File path for checkerboard image must be provided for image-based calibration.")
                # Load image
                img = cv2.imread(file_path_image)
                if img is None:
                    raise ValueError(f"Could not load image from {file_path_image}.")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            case "live":
                # Get a live color frame
                input("Place the checkerboard in front of the camera and press Enter to capture a frame for calibration...")
                img = self.get_live_frame(stream=rs.stream.color)
                if img is None:
                    raise ValueError("Could not capture a color frame for calibration.")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        objp = np.zeros((cam_config.CHECKERBOARD_SIZE[0] * cam_config.CHECKERBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:cam_config.CHECKERBOARD_SIZE[0], 0:cam_config.CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
        objp = objp * cam_config.SQUARE_SIZE_M

        corners_are_found, corners = cv2.findChessboardCorners(gray, cam_config.CHECKERBOARD_SIZE, None)
        if not corners_are_found:
            raise RuntimeError("Could not find checkerboard corners. Ensure the checkerboard is fully visible and well-lit.")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        _, rvec, tvec = cv2.solvePnP(objp, corners_refined, self.K_matrix, self.D_matrix)

        self.R_matrix, _ = cv2.Rodrigues(rvec)
        self.T_vector = tvec.flatten()

        # --- Save the extrinsic matrices for future use ---
        with open(cam_config.CALIBRATION_T_MATRIX_FILEPATH, 'w') as f_T:
            np.savetxt(f_T, self.T_vector)
        with open(cam_config.CALIBRATION_R_MATRIX_FILEPATH, 'w') as f_R:
            np.savetxt(f_R, self.R_matrix)
        return

    def get_live_frame(self, stream=rs.stream.color) -> np.ndarray | None:
        """Captures and returns the latest frame for a specified stream.
        
        Args:
            stream: The RealSense stream type (rs.stream.color or rs.stream.depth).
        
        Returns:
            np.ndarray | None: The captured frame as a numpy array, or None if no frame is available.
        """
        # Wait for a coherent set of frames (usually takes a few frames to stabilize)
        for i in range(10): # Flush pipeline with 10 frames
            frames = self.pipeline.wait_for_frames()

        if stream == rs.stream.color:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            # Convert frame to numpy array
            return np.asanyarray(color_frame.get_data())
        
        if stream == rs.stream.depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            # Convert frame to numpy array
            return np.asanyarray(depth_frame.get_data())
        
    def save_frame(self, frame: np.ndarray, file_path: str = f"data/frame_|_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png") -> None:
        """Saves a given frame to an image file.
        
        Args:
            frame (np.ndarray): The frame to save.
            file_path (str): The file path where to save the image. Default location is "data/frame_<timestamp>.png".

        Returns:
            None
        """
        cv2.imwrite(file_path, frame)
        return