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
    
    def _create_pixel_to_real_world_map(self, mode: str = "live") -> None:
        """
        Sets the pixel_to_real_world_conversion_info attribute by detecting the centroids of the lines in the calibration image and categorizing them into N, E, S, W based on their position relative to the center of the image. The reference point in pixel coordinates is also stored for later use in conversion.

        Args:
            mode (str): "live" to use a live frame for calibration, "image" to use a pre-captured image for calibration. (default: "live")

        Returns:
            None
        """
        match mode:
            case "live":
                img = self.get_frame(stream=rs.stream.color)
                if img is None:
                    raise ValueError("Could not capture a color frame for creating pixel to real world map.")
            case "image":
                img = cv2.imread(cam_config.CALIBRATION_REAL_WORLD_PATH)
                if img is None:
                    raise ValueError(f"Could not load image from {cam_config.CALIBRATION_REAL_WORLD_PATH} for creating pixel to real world map.")
            case _:
                raise ValueError("Invalid mode. Choose 'live' or 'image'.")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        blur_gray = cv2.GaussianBlur(equalized,(3, 3),0)
        edges = cv2.Canny(blur_gray, cam_config.CANNY_LOW_THRESHOLD, cam_config.CANNY_HIGH_THRESHOLD)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # Draw contours for visualization
        cv2.imshow("Contours", img)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # Enter to continue after visualizing contours
                cv2.destroyWindow("Contours")
                break
            if key == ord('r'): # Press 'R' to reprocess the image if contours are not satisfactory
                cv2.destroyWindow("Contours")
                return self._create_pixel_to_real_world_map(mode=mode)


        centroids = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h != 0 else 0
            if 1 < max(w, h) < 100 and min(w, h) < 20:  # Filter contours based on size to find line segments
                if any(np.hypot((x + w // 2) - ex, (y + h // 2) - ey) < 10 and w>h == horizontal for ex, ey, horizontal in centroids):
                    continue  # Skip if this contour is close to an already detected centroid
                centroids.append((x + w // 2, y + h // 2, w>h))
            elif 0.2 < aspect_ratio < 2 and w > 10 and h > 5: # Filter the reference square
                real_world_reference_p = (x, y) # Real world reference point in pixel coordinates
        
        # calculate the center of all the centroids to know where each centroid is located in the image (N, E, S, W)
        center_x = np.mean([c[0] for c in centroids])
        center_y = np.mean([c[1] for c in centroids])

        # show the centroids and the center point for visualization
        for c in centroids:
            cv2.circle(img, (c[0], c[1]), 2, (255, 0, 0), -1)
        cv2.circle(img, (int(center_x), int(center_y)), 2, (0, 0, 255), -1)
        cv2.circle(img, real_world_reference_p, 5, (0, 255, 255), -1) # Reference point in yellow
        print(f"Reference point: {real_world_reference_p}")
        cv2.imshow("Centroids and Center", img)
        while True:            
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # Enter to continue after visualizing centroids and center
                cv2.destroyWindow("Centroids and Center")
                break
            if key == ord('r'): # Press 'R' to reprocess the image if centroids are not satisfactory
                cv2.destroyWindow("Centroids and Center")
                return self._create_pixel_to_real_world_map(mode=mode)

        NESW_info = {"N": [], "E": [], "S": [], "W": []}
        for centroid in centroids:
            if centroid[2]:  # Vertical line segment
                if centroid[1] < center_y:
                    NESW_info["N"].append(centroid)
                else:
                    NESW_info["S"].append(centroid)
            else:  # Horizontal line segment
                if centroid[0] < center_x:
                    NESW_info["W"].append(centroid)
                else:
                    NESW_info["E"].append(centroid)
        
        # Sort the centroids
        for direction in NESW_info:
            NESW_info[direction].sort(key=lambda c: c[1] if direction in ["N", "S"] else c[0])

        # Store the x and y values
        if len(NESW_info["N"]) != len(NESW_info["S"]) or len(NESW_info["E"]) != len(NESW_info["W"]):
            # raise ValueError("Unequal number of centroids detected in opposite directions while creating pixel to real world map. Ensure the lines are fully visible and well-detected.")
            pass

        self.pixel_to_real_world_conversion_info = {"x": [], "y": [], "reference_pixel": real_world_reference_p}
        for N_point, S_point in zip(NESW_info["N"], NESW_info["S"]):
            self.pixel_to_real_world_conversion_info["x"].append(np.mean([N_point[0], S_point[0]]))
        for E_point, W_point in zip(NESW_info["E"], NESW_info["W"]):
            self.pixel_to_real_world_conversion_info["y"].append(np.mean([E_point[1], W_point[1]]))

        # write to json file
        if mode == "live":
            os.makedirs(os.path.dirname(cam_config.CALIBRATION_REAL_WORLD_PATH), exist_ok=True)
            with open(cam_config.CALIBRATION_REAL_WORLD_PATH, "w") as f:
                json.dump(self.pixel_to_real_world_conversion_info, f, indent=4)
        
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
            mode (str): "live", "load", or "image" to specify how to set up R and T. (default: "live")
            file_path_T (str): File path to load the T vector from (required if mode="load").
            file_path_R (str): File path to load the R matrix from (required if mode="load").      
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
        
            case "live":
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
        self.pixel_to_real_world_map = self._create_pixel_to_real_world_map(mode=mode)
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
        Converts pixel coordinates to real world coordinates using the current calibration and the pixel_to_real_world_conversion_info.

        Args:
            pixel_coords (tuple[int, int]): The pixel coordinates (x, y) to be converted.

        Returns:
            tuple[float, float]: The corresponding real world coordinates (x, y).
        """

        def interpolate_axis(pixel_val: int, grid: list[int], reference_p: int) -> float:
            """
            Helper function to interpolate the real world coordinate along one axis based on the pixel value, the grid of pixel values for that axis, and the reference pixel value.
            
            Args:                
                pixel_val (int): The pixel coordinate value along the axis to be converted.
                grid (list[int]): The list of pixel values along the axis corresponding to the real world grid lines.
                reference_p (int): The reference pixel value for interpolation."""
            # ignore lines before the reference point
            grid = [g for g in grid if g >= reference_p]
            for i in range(len(grid) - 1):
                if grid[i] <= pixel_val <= grid[i+1]:
                    gap_pixels = grid[i+1] - grid[i]
                    fraction = (pixel_val - grid[i]) / gap_pixels if gap_pixels != 0 else 0.0
                    return (i + fraction) * cam_config.DISTANCE_BETWEEN_LINES_MM
            raise ValueError(f"Pixel value {pixel_val} is out of bounds of the grid.")
        
        if self.pixel_to_real_world_conversion_info is None:
            raise ValueError("Pixel to real world conversion info not set. Call setup_matrices() first.")
        
        target_x_p, target_y_p = pixel_coords
        reference_x_p, reference_y_p = self.pixel_to_real_world_conversion_info["reference_pixel"]
        x_grid_p = self.pixel_to_real_world_conversion_info["x"]
        y_grid_p = self.pixel_to_real_world_conversion_info["y"]

        target_x_w = interpolate_axis(target_x_p, x_grid_p, reference_x_p)
        target_y_w = interpolate_axis(target_y_p, y_grid_p, reference_y_p)

        return target_x_w, target_y_w