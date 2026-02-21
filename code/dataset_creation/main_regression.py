import os
import cv2
import pyrealsense2 as rs
import json

import dataset_creation.config as config
from camera.Camera import RealSenseCamera

current_center: tuple[float, float] = None

def mark_center(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global current_center
        current_center = (x, y)

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
                # Prepare for annotation
                temp_image = color_image.copy()
                current_center = None
                cv2.namedWindow("Annotate")
                cv2.setMouseCallback("Annotate", mark_center)
                
                
                print(f"\n--- Annotation Mode ---")
                print("\t1. Click on image to set center.")
                print("\t2. Press [S] to SAVE image + txt.")
                print("\t3. Press [R] to RECAPTURE (discard).")
                print("\t4. Press [Z] to Undo the center.")

                while True:
                    # Create a clean copy to indicate the centers on
                    display_img = temp_image.copy()
                    if current_center:
                        cv2.drawMarker(display_img, current_center, (255, 0, 0), 
                                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2,)
                    
                    # Update the window to show new or removed markers
                    cv2.imshow("Annotate", display_img)
                    k2 = cv2.waitKey(1) & 0xFF

                    # [S] SAVE
                    if k2 == ord('s'):
                        # 1. Save Image
                        img_filename = f"{config.IMAGE_ID_START:07}.png"
                        json_filename = f"{config.IMAGE_ID_START:07}.json"
                        config.IMAGE_ID_START += 1
                        
                        img_path = os.path.join(config.ROOT_DIRECTORY, img_filename)
                        json_path = os.path.join(config.ROOT_DIRECTORY, json_filename)
                        
                        camera.save_frame(color_image, depth_image, file_path=img_path)
                        
                        # 2. Save Center coordinates (JSON)
                        with open(json_path, "w") as f:
                            json.dump({
                                "pixel": {
                                    "x": current_center[0],
                                    "y": current_center[1]
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