import os
import cv2
import pyrealsense2 as rs
import easygui

import dataset_creation.config as config
from camera.Camera import RealSenseCamera


current_label: str = None
current_boxes: list = []
drawing: bool = False
ix, iy = -1, -1 # Initial x, y (where you clicked down)
cx, cy = -1, -1 # Current x, y (where your mouse is dragging)




def mouse_callback(event, x, y, flags, param) -> None:
    """
    Handles mouse events to draw bounding boxes.
    
    Inputs:
        event: The mouse event
        x, y: The current mouse coordinates
        flags: Any relevant flags passed by OpenCV
        param: Additional parameters (not used here)
        
    Returns:
        None (modifies global state)
    """
    global ix, iy, cx, cy, drawing, current_boxes

    # Click to start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cx, cy = x, y # Initialize current to start

    # Dragging the mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update current position continuously while dragging
            cx, cy = x, y

    # Release to finish drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Ensure coordinates are top-left to bottom-right
        x_min, x_max = sorted([ix, x])
        y_min, y_max = sorted([iy, y])
        
        # Only add if it has some area
        if x_max > x_min and y_max > y_min:
            current_boxes.append((x_min, y_min, x_max, y_max))

def convert_to_yolo(box, img_width, img_height) -> tuple:
    """
    Converts pixel box (xmin, ymin, xmax, ymax) to YOLO (x_c, y_c, w, h).
    
    inputs:
        box: tuple of (xmin, ymin, xmax, ymax) in pixels
        img_width: width of the image in pixels
        img_height: height of the image in pixels
        
    returns:
        tuple of (x_center, y_center, width, height) normalized [0, 1]
    """
    xmin, ymin, xmax, ymax = box
    
    # Calculate center, width, and height
    bw = xmax - xmin
    bh = ymax - ymin
    bx = xmin + bw / 2.0
    by = ymin + bh / 2.0
    
    # Normalize by image dimensions
    return (bx / img_width, by / img_height, bw / img_width, bh / img_height)

def main():
    global current_label

    if config.IMAGE_ID_START == 0:
        print("Starting image IDs from 0. CHANGE THIS IF YOU ALREADY CREATED SOME IMAGES TO AVOID OVERWRITING!")

    print(f"Images will be stored in: {config.ROOT_DIRECTORY}\nControls:\n\t[C] to capture a frame.\n\t[L] to change label.\n\t[ESC] to quit.")

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

        # User presses [L] to change label or if no label is set. Let them choose one and make sure it is aligned with the dataset
        if current_label is None or key == ord('l'):
            print("Choose a label from the popup window.")
            choice = easygui.choicebox("Pick a label", "Label picker", list(config.CLASSES.keys()))
            if choice is not None:
                # Ensure directory exists
                if os.path.exists(os.path.join(config.ROOT_DIRECTORY, choice)) is False:
                    os.makedirs(os.path.join(config.ROOT_DIRECTORY, choice))

                # Set current label
                current_label = choice
                print(f"Current label set to: {current_label}")
            print("Controls:\n\t[C] to capture a frame.\n\t[L] to change label.\n\t[ESC] to quit.")

        # User presses [C] to capture and annotate current frame
        if key == ord('c'):
            global current_boxes
            if color_image is not None:
                # Prepare for annotation
                temp_image = color_image.copy()
                current_boxes = [] # Reset boxes for this new frame
                cv2.namedWindow("Annotate")
                cv2.setMouseCallback("Annotate", mouse_callback)
                
                print(f"\n--- Annotation Mode ({current_label}) ---")
                print("\t1. Drag mouse to draw bounding boxes.")
                print("\t2. Press [S] to SAVE image + txt.")
                print("\t3. Press [R] to RECAPTURE (discard).")
                print("\t4. Press [Z] to Undo last box.")

                while True:
                    # Create a clean copy to draw on
                    display_img = temp_image.copy()
                    
                    # Draw existing boxes (Green)
                    for box in current_boxes:
                        cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                    if drawing:
                        cv2.rectangle(display_img, (ix, iy), (cx, cy), (0, 0, 255), 2)

                    cv2.imshow("Annotate", display_img)
                    k2 = cv2.waitKey(1) & 0xFF

                    # [S] SAVE
                    if k2 == ord('s'):
                        # 1. Save Image
                        img_filename = f"{config.IMAGE_ID_START:07}.png"
                        txt_filename = f"{config.IMAGE_ID_START:07}.txt"
                        config.IMAGE_ID_START += 1
                        
                        img_path = os.path.join(config.ROOT_DIRECTORY, current_label, img_filename)
                        txt_path = os.path.join(config.ROOT_DIRECTORY, current_label, txt_filename)
                        
                        camera.save_frame(color_image, depth_image, file_path=img_path)
                        
                        # 2. Save Labels (TXT)
                        if current_boxes:
                            class_id = config.CLASSES[current_label]
                            h, w, _ = color_image.shape
                            
                            with open(txt_path, 'w') as f:
                                for box in current_boxes:
                                    # Convert to YOLO format
                                    yolo_bbox = convert_to_yolo(box, w, h)
                                    # Write: class_id x_center y_center width height
                                    f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                            print(f"Saved: {img_filename} with {len(current_boxes)} labels.")
                        else:
                            print(f"Saved: {img_filename} (No labels drawn).")
                            
                        cv2.destroyWindow("Annotate")
                        break

                    # [R] RECAPTURE
                    elif k2 == ord('r'):
                        print("Discarded.")
                        cv2.destroyWindow("Annotate")
                        break
                    
                    # [Z] UNDO
                    elif k2 == ord('z'):
                        if current_boxes:
                            current_boxes.pop()
                            print("Removed last box.")
            print("Controls:\n\t[C] to capture a frame.\n\t[L] to change label.\n\t[ESC] to quit.")

        if key == 27:  # Press 'Esc' to exit
            print("Exiting...")
            break


    # Stop the camera stream
    camera.stop_stream()

if __name__ == "__main__":
    main()