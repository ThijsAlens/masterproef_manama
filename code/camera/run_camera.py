"""
This file contains an example of how to use the Camera class to capture and display
"""

from camera.Camera import RealSenseCamera
import pyrealsense2 as rs
import cv2


def main():
    camera = RealSenseCamera()
    camera.start_stream()
    
    is_calibrated = False
    print_calibration_prompt = True
    
    while True:
        color_image = camera.get_frame(stream=rs.stream.color)
        depth_image = camera.get_frame(stream=rs.stream.depth)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)

        key = cv2.waitKey(1) & 0xFF

        if not is_calibrated:
            if print_calibration_prompt:
                print("Place the checkerboard in front of the camera and press Enter to capture a frame for calibration...")
                print_calibration_prompt = False
            
            if key == 13:  # Check for Enter key press
                try:
                    print("Attempting calibration...")
                    camera.setup_matrices(mode="live")
                    
                    print("Camera calibrated successfully.")
                    is_calibrated = True
                    
                except Exception as e:
                    print(f"Calibration failed: {e}")

        if key == 27:  # Press 'Esc' to exit
            print("Exiting stream...")
            break

if __name__ == "__main__":
    main()