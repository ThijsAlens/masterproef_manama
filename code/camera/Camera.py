import pyrealsense2 as rs
import numpy as np
import cv2

import threading
import time

class Camera:
    """
    A class that defines an interface for a camera.

    It uses a background thread to continuously capture frames from the camera. To read the latest frame, call the get_frames() method.
    """
    def __init__(self, colour_resolution: tuple[int, int]=(1920, 1080), depth_resolution: tuple[int, int]=(1280, 720), fps: int=30, enable_colour: bool=True, enable_depth: bool=True, show_colour: bool=True, show_depth: bool=True):
        # Camera configuration
        self.color_resolution = colour_resolution
        self.depth_resolution = depth_resolution
        self.fps: int = fps

        # Configure what to capture and display
        self.enable_colour: bool = enable_colour
        self.enable_depth: bool = enable_depth
        self.capture_thread: threading.Thread = None
        self.show_colour: bool = show_colour
        self.show_colour_thread: threading.Thread = None
        self.show_depth: bool = show_depth
        self.show_depth_thread: threading.Thread = None

        # Initialize camera resources
        self.pipeline: rs.pipeline = None
        self.config: rs.config = None
        self.running: bool = False

        # Frame storage
        self.frame_lock = threading.Lock()
        self.latest_colour_frame: np.ndarray = None
        self.latest_depth_frame: np.ndarray = None
    
    # ------------------------------------------------------------
    # Internal Methods (threads)
    # ------------------------------------------------------------
    def _capture_loop(self):
        """Background thread that constantly reads frames."""
        while self.running:
            frames = self.pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame() if self.enable_depth else None
            color_frame = frames.get_color_frame() if self.enable_colour else None

            with self.frame_lock:
                if depth_frame:
                    self.latest_depth = np.asanyarray(depth_frame.get_data())
                if color_frame:
                    self.latest_color = np.asanyarray(color_frame.get_data())

    def _display_colour_loop(self):
        """Background thread to display color frames."""
        while self.running:
            with self.frame_lock:
                if self.latest_colour_frame is not None:
                    cv2.imshow("Color Frame", self.latest_colour_frame)
            cv2.waitKey(1)
            
    def _display_depth_loop(self, enable_colour: bool=False):
        """Background thread to display depth frames."""
        while self.running:
            with self.frame_lock:
                if self.latest_depth_frame is not None:
                    if enable_colour:
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.latest_depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                    else:
                        depth_colormap = cv2.convertScaleAbs(self.latest_depth_frame, alpha=0.03)
                    cv2.imshow("Depth Frame", depth_colormap)
            cv2.waitKey(1)

    # ------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------
    def start(self):
        """Starts the RealSense camera streaming in a background thread."""
        if self.running:
            return

        # Configure RealSense streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        if self.enable_colour:
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Start streaming
        self.pipeline.start(self.config)
        self.running = True

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # Start display threads if needed
        if self.show_colour:
            self.show_colour_thread = threading.Thread(target=self._display_colour_loop, daemon=True)
            self.show_colour_thread.start()
        if self.show_depth:
            self.show_depth_thread = threading.Thread(target=self._display_depth_loop, daemon=True, args=(False,))
            self.show_depth_thread.start()

    def stop(self):
        """Stops the camera streaming."""
        if not self.running:
            return
        self.running = False
        time.sleep(0.1)  # small delay to allow thread to exit

        if self.pipeline:
            self.pipeline.stop()

        self.pipeline = None
        self.config = None
        self.thread = None

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the most recent color and depth frames.
        Returns (color, depth)
        Either may be None if not enabled or not received yet.
        """
        with self.frame_lock:
            color = self.latest_color.copy() if self.latest_color is not None else None
            depth = self.latest_depth.copy() if self.latest_depth is not None else None
        return color, depth
