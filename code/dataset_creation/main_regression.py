import os
import cv2
import pyrealsense2 as rs
import json

import dataset_creation.config as config
from camera.Camera import RealSenseCamera
from dataset_creation.calculate_real_world_coordinates import calculate_real_world_coordinates

current_center: tuple[float, float] = None

def mark_center(event, x, y, flags, param):
    """Manual override: Clicking sets the center manually."""
    if event == cv2.EVENT_LBUTTONDOWN:
        global current_center
        current_center = (x, y)

def auto_detect_disk_center(image) -> tuple[int, int]:
    """
    Uses contour detection to automatically find circle-like disks.
    Returns the (x, y) coordinates of the centroid, or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_center = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 1. Filter out noise
        if area < 100:  
            continue
            
        # 2. Filter by Aspect Ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        if 0.7 < aspect_ratio < 1.3:
            # If multiple circle-like things are found, grab the biggest one
            if area > max_area:
                max_area = area
                
                # cv2.moments calculates the true center of mass of the shape
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_center = (cx, cy)
                    
    return best_center

def main():
    global current_center

    if config.IMAGE_ID_START == 0:
        print("Starting image IDs from 0. CHANGE THIS IF YOU ALREADY CREATED SOME IMAGES TO AVOID OVERWRITING!")

    print(f"Images will be stored in: {config.ROOT_DIRECTORY}\nControls:\n\t[C] to capture a frame.\n\t[ESC] to quit.")

    # Initialize the camera
    camera = RealSenseCamera()
    camera.start_stream()
    camera.setup_matrices(mode="live")

    while True:
        # Capture and show frames
        color_image = camera.get_frame(stream=rs.stream.color)
        depth_image = camera.get_frame(stream=rs.stream.depth)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF

        # User presses [C] to capture and annotate current frame
        if key == ord('c'):
            if color_image is not None:
                temp_image = color_image.copy()
                
                # --- AUTO DETECT TRIGGER ---
                # Attempt to find the center automatically before opening the window
                current_center = auto_detect_disk_center(temp_image)
                
                cv2.namedWindow("Annotate")
                cv2.setMouseCallback("Annotate", mark_center)
                
                print(f"\n--- Annotation Mode ---")
                if current_center:
                    print("--> Disk auto-detected! Press [S] to confirm.")
                else:
                    print("--> Auto-detect failed. Click the disk to set center manually.")
                print("\t1. Click on image to override/set center.")
                print("\t2. Press [S] to SAVE image + txt.")
                print("\t3. Press [R] to RECAPTURE (discard).")
                print("\t4. Press [Z] to Undo the center.")

                while True:
                    # Create a clean copy to indicate the centers on
                    display_img = temp_image.copy()
                    if current_center:
                        cv2.drawMarker(display_img, current_center, (0, 255, 0), 
                                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                    
                    # Update the window to show new or removed markers
                    cv2.imshow("Annotate", display_img)
                    k2 = cv2.waitKey(1) & 0xFF

                    # [S] SAVE
                    if k2 == ord('s'):
                        if current_center is None:
                            print("No center marked! Please click to mark the center before saving.")
                            continue
                            
                        # 1. Save Image
                        img_filename = f"{config.IMAGE_ID_START:07}.png"
                        json_filename = f"{config.IMAGE_ID_START:07}.json"
                        config.IMAGE_ID_START += 1
                        
                        img_path = os.path.join(config.ROOT_DIRECTORY, img_filename)
                        json_path = os.path.join(config.ROOT_DIRECTORY, json_filename)
                        
                        camera.save_frame(color_image, depth_image, file_path=img_path)
                        
                        # 2. Calculate real-world coordinates
                        x_w, y_w = camera.convert_pixel_to_real_world(current_center)

                        # 3. Save Center coordinates (JSON)
                        with open(json_path, "w") as f:
                            json.dump({
                                "pixel": {
                                    "x": current_center[0],
                                    "y": current_center[1]
                                },
                                "world": {
                                    "x": x_w,
                                    "y": y_w
                                }
                            }, f, indent=4)
                            
                        cv2.destroyWindow("Annotate")
                        break

                    # [R] RECAPTURE
                    elif k2 == ord('r'):
                        print("Discarded.")
                        cv2.destroyWindow("Annotate")
                        break
                    
                    # [Z] UNDO
                    elif k2 == ord('z'):
                        if current_center:
                            current_center = None
                            print("Removed last center.")
            print("Controls:\n\t[C] to capture a frame.\n\t[ESC] to quit.")

        if key == 27:  # Press 'Esc' to exit
            print("Exiting...")
            break

    # Stop the camera stream
    camera.stop_stream()

if __name__ == "__main__":
    main()