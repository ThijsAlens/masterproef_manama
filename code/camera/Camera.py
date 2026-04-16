import datetime
import json
import os

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
        self.mapx = None                # Undistortion map for x-coordinates, used for remapping the image to correct distortion
        self.mapy = None                # Undistortion map for y-coordinates, used for remapping the image to correct distortion
        self.rs_intrinsics = None       # RealSense Intrinsics object, used for the translation from pixel to camera coordinates
        self.R_matrix = None            # Rotation Matrix (World to Camera)
        self.T_vector = None            # Translation Vector (World to Camera)
        self.pixel_to_real_world_conversion_info = None  # This structure will hold the information needed to convert pixel coordinates to real world coordinates

        # RealSense pipeline setup
        self.pipeline = rs.pipeline()   # Create a RealSense pipeline, this is used to configure, start and stop the camera
        self.config = rs.config()       # Create a config object to configure the pipeline
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)   # Enable depth stream
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # Enable color stream        
        return

    def start_stream(self):
        """
        Starts the RealSense pipeline and get the intrinsics.

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
    
    def setup_matrices(self, mode: str = "live", file_path_T: str = None, file_path_R: str = None) -> None:
        """
        Sets up the intrinsic (K, D) and extrinsic (R, T) matrices for the camera.
        Intrinsic matrices are calculated using the RealSense intrinsics, if they are not set, run start_stream() first.
        Extrinsic matrices (R and T) are:
            - loaded from a previous run (specify file paths) (mode="load")
            - calculated from live calibration (needs the pipeline, if not active, run start_stream()) (mode="live", default)
            - calculated from a checkerboard image (setup_live=False, specify file path) (mode="image")

        Args:
            mode (str): "live", "load", "image", or "setup" to specify how to set up R and T. (default: "live")
            file_path_T (str): File path to load the T vector from (required if mode="load").
            file_path_R (str): File path to load the R matrix from (required if mode="load").      
        Returns:
            None
        """

        if self.rs_intrinsics is None:
            raise ValueError("RealSense intrinsics not set. Call start_stream() first.")
        if mode != "live" and mode != "load" and mode != "image" and mode != "setup":
            raise ValueError("Invalid mode. Choose 'live', 'load', 'image', or 'setup'.")
        
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
                # Load T and R matrices from specified file paths
                self.T_vector = np.loadtxt(file_path_T)
                self.R_matrix = np.loadtxt(file_path_R)
                return

            case "image":
                # Load image
                img = cv2.imread(cam_config.CALIBRATION_IMAGE_PATH)
                if img is None:
                    raise ValueError(f"Could not load image from {cam_config.CALIBRATION_IMAGE_PATH}.")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            case "live" | "setup":
                # Get a live color frame for calibration
                img = self.get_frame(stream=rs.stream.color)
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
        os.makedirs(os.path.dirname(cam_config.CALIBRATION_T_MATRIX_FILEPATH), exist_ok=True)
        os.makedirs(os.path.dirname(cam_config.CALIBRATION_R_MATRIX_FILEPATH), exist_ok=True)
        
        with open(cam_config.CALIBRATION_T_MATRIX_FILEPATH, 'w') as f_T:
            np.savetxt(f_T, self.T_vector)
        with open(cam_config.CALIBRATION_R_MATRIX_FILEPATH, 'w') as f_R:
            np.savetxt(f_R, self.R_matrix)

        # --- Set up the pixel to real world coordinate mapping for the current calibration ---
        if mode == "live":
            input("Change to the lines to calibrate the pixel to real world mapping and press Enter to continue...")
        
        # Load the pixel to real world conversion info from the specified file path
        if mode == "setup":
            # This mode is used to set up the pixel to real world conversion info based on the lines image, no further action should be taken with this instance of the camera.
            return
        with open(cam_config.CALIBRATION_MAPPING_FILEPATH, 'r') as f:
            self.pixel_to_real_world_conversion_info = json.load(f)
        return

    def get_frame(self, stream=rs.stream.color, straighten=True, crop: bool = False) -> np.ndarray | None:
        """
        Captures and returns the latest frame for a specified stream.
        
        Args:
            stream: The RealSense stream type (rs.stream.color or rs.stream.depth).
            straighten: Whether to straighten the image using undistortion maps. (default: True)
            crop: If set to True, the frame will exclude the measuring lines (default: False)

        Returns:
            np.ndarray | None: The captured frame as a numpy array, or None if no frame is available.
        """
        
        frames = self.pipeline.wait_for_frames()

        if stream == rs.stream.color:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            # Convert frame to numpy array
            img = np.asanyarray(color_frame.get_data())
            if straighten:
                img, _ = self.straighten_frame(img, None)
            
            # Default cropping
            img = img[cam_config.DEFAULT_CROP_Y[0]:cam_config.DEFAULT_CROP_Y[1], cam_config.DEFAULT_CROP_X[0]:cam_config.DEFAULT_CROP_X[1]]
            
            if crop: # Crop the image to exclude the measuring lines based
                img = img[cam_config.EXCLUDE_MEASURING_Y[0]:cam_config.EXCLUDE_MEASURING_Y[1], cam_config.EXCLUDE_MEASURING_X[0]:cam_config.EXCLUDE_MEASURING_X[1]]
            return img
        
        if stream == rs.stream.depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            # Convert frame to numpy array
            img = np.asanyarray(depth_frame.get_data())
            if straighten:
                _, img = self.straighten_frame(None, img)
            
            # Default cropping
            img = img[cam_config.DEFAULT_CROP_Y[0]:cam_config.DEFAULT_CROP_Y[1], cam_config.DEFAULT_CROP_X[0]:cam_config.DEFAULT_CROP_X[1]]
            if crop: # Crop the image to exclude the measuring lines based
                img = img[cam_config.EXCLUDE_MEASURING_Y[0]:cam_config.EXCLUDE_MEASURING_Y[1], cam_config.EXCLUDE_MEASURING_X[0]:cam_config.EXCLUDE_MEASURING_X[1]]
            return img
        
    def save_frame(self, color_image: np.ndarray, depth_image: np.ndarray, file_path: str = None) -> str:
        """
        Saves a given frame to an image file.
        
        Args:
            color_image (np.ndarray): The color image to save.
            depth_image (np.ndarray): The depth image to save.
            file_path (str): The file path where to save the image. Default location is "data/frame_<timestamp>_<type>.png". Example: when "image" is given as filename, it will be saved as 'data/image_color.png' and 'data/image_depth.png'.

        Returns:
            str: The file path where the image was saved.
        """
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            os.makedirs("data", exist_ok=True)
            color_file_path = f"data/frame_{timestamp}_color.png"
            depth_file_path = f"data/frame_{timestamp}_depth.png"
        else:
            color_file_path = file_path.replace(".png", "_color.png")
            depth_file_path = file_path.replace(".png", "_depth.png")

        cv2.imwrite(color_file_path, color_image)
        cv2.imwrite(depth_file_path, depth_image)
        return color_file_path, depth_file_path
    
    def straighten_frame(self, color_img: np.ndarray = None, depth_img: np.ndarray = None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Applies undistortion to the given image using the camera's intrinsic parameters.

        Args:
            color_img (np.ndarray): The color image to be straightened. (default: None, if only depth image is to be straightened)
            depth_img (np.ndarray): The depth image to be straightened. (default: None, if only color image is to be straightened)

        Returns:
            tuple[np.ndarray, np.ndarray | None]: The undistorted (straightened) color and depth images.
        """
        if self.K_matrix is None or self.D_matrix is None:
            raise ValueError("Camera intrinsics not set. Call setup_matrices() first.")
        
        if color_img is None and depth_img is None:
            raise ValueError("At least one of color_img or depth_img must be provided for straightening.")
        
        if color_img is not None:
            h, w = color_img.shape[:2]
        elif depth_img is not None:
            h, w = depth_img.shape[:2]
            
        new_K, roi = cv2.getOptimalNewCameraMatrix(self.K_matrix, self.D_matrix, (w, h), 1, (w, h))
        if self.mapx is None or self.mapy is None:
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.K_matrix, self.D_matrix, None, new_K, (w, h), cv2.CV_32FC1)
        corrected_color_img = cv2.remap(color_img, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR) if color_img is not None else None
        corrected_depth_img = cv2.remap(depth_img, self.mapx, self.mapy, interpolation=cv2.INTER_NEAREST) if depth_img is not None else None

        x, y, w, h = roi
        undistorted_color_img = corrected_color_img[y:y+h, x:x+w] if corrected_color_img is not None else None
        undistorted_depth_img = corrected_depth_img[y:y+h, x:x+w] if corrected_depth_img is not None else None

        return (undistorted_color_img, undistorted_depth_img)
    
    def convert_pixel_to_real_world(self, pixel_coords: tuple[int, int]) -> tuple[float, float]:
        """
        Converts pixel coordinates to real-world coordinates using a grid-search 
        and linear interpolation algorithm.
        """
        if getattr(self, 'pixel_to_real_world_conversion_info', None) is None:
            raise ValueError("Calibration info not set. Call load_calibration_map() first.")
            
        info = self.pixel_to_real_world_conversion_info
        grid = info["grid_2d"]
        ref_px = info["reference_pixel"]
        
        target_px = pixel_coords
        gap_mm = float(cam_config.DISTANCE_BETWEEN_LINES_MM)

        def get_linear_mm(px, py):
            """Executes the grid-search and linear interpolation for a single pixel."""
            rows = len(grid)
            cols = len(grid[0])
            
            # --- 1. Find the Row (Y) ---
            target_row = 0
            for r in range(rows):
                # Search for the lowest y-value in this row
                min_y = min(pt[1] for pt in grid[r])
                
                # If the row's lowest Y exceeded our target Y, we stepped too far down!
                if min_y > py:
                    target_row = max(0, r - 1)  # The target is in the previous row
                    break
                target_row = r  # Keep updating in case it's in the very last row
            
            # Cap it so we don't crash if clicked below the grid
            target_row = min(target_row, rows - 2)
            
            # --- 2. Find the Column (X) ---
            target_col = 0
            for c in range(cols):
                # Find the lowest X in this column (checking top and bottom corners of the row we found)
                min_x = min(grid[target_row][c][0], grid[target_row+1][c][0])
                
                if min_x > px:
                    target_col = max(0, c - 1) # Target is in the previous column
                    break
                target_col = c
                
            # Cap it so we don't crash if clicked to the right of the grid
            target_col = min(target_col, cols - 2)
            
            # --- 3. Linear Interpolation ---
            tl = grid[target_row][target_col]       # Top-Left corner of cell
            tr = grid[target_row][target_col + 1]   # Top-Right corner of cell
            bl = grid[target_row + 1][target_col]   # Bottom-Left corner of cell
            
            # Calculate the width and height of this specific cell in pixels
            cell_width_px = tr[0] - tl[0]
            cell_height_px = bl[1] - tl[1]
            
            # Calculate the percentage (0.0 to 1.0) of how far the pixel is inside the cell
            pct_x = (px - tl[0]) / cell_width_px if cell_width_px != 0 else 0.0
            pct_y = (py - tl[1]) / cell_height_px if cell_height_px != 0 else 0.0
            
            # --- 4. Final Math ---
            # Rough grid position + interpolated sub-cell position
            real_x = (target_col + pct_x) * gap_mm
            real_y = (target_row + pct_y) * gap_mm
            
            return real_x, real_y

        # Map the Target and the Reference through the algorithm
        abs_target_x, abs_target_y = get_linear_mm(target_px[0], target_px[1])
        abs_ref_x, abs_ref_y = get_linear_mm(ref_px[0], ref_px[1])

        # Subtract Reference to lock the square at (0, 0)
        final_x = abs_target_x - abs_ref_x
        final_y = abs_target_y - abs_ref_y

        return float(final_x), float(final_y)