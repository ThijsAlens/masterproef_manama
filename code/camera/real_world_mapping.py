import cv2
import numpy as np
import json
import os
import pyrealsense2 as rs

from camera.Camera import RealSenseCamera
import camera.config as cam_config

def get_line_intersection(line1, line2):
    """Calculates the exact (x, y) intersection of two lines defined by two points each."""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b): return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0: raise Exception('Lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [float(x), float(y)]

def main():
    print("Starting Manual 2D Mesh Calibration...")
    
    # 1. Initialize Camera
    camera = RealSenseCamera()
    camera.start_stream()
    camera.setup_matrices(mode="setup")
    input("READY?")
    
    try:
        # Grab a single frame to draw on
        img = camera.get_frame(stream=rs.stream.color)
        if img is None:
            raise ValueError("Could not capture a color frame.")
            
        # 2. GUI State Variables
        points = {"N": [], "S": [], "E": [], "W": [], "REF": []}
        state = {"mode": "REF"} 
        colors = {"N": (255, 0, 0), "S": (0, 0, 255), "E": (0, 255, 0), "W": (255, 0, 255), "REF": (0, 255, 255)}

        # 3. Mouse Click Event (Pure manual placement)
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                current_mode = state["mode"]
                if current_mode == "REF": 
                    points["REF"] = [(x, y)]
                else: 
                    points[current_mode].append((x, y))
                print(f"--> Placed point {len(points[current_mode])} in [{current_mode}] at ({x}, {y})")

        # 4. Setup Window
        cv2.namedWindow("Create 2D Mesh")
        cv2.setMouseCallback("Create 2D Mesh", mouse_callback)

        print("\n" + "="*40)
        print("--- PURE MANUAL CALIBRATION ---")
        print("Controls:")
        print("[1] Ref Square  [2] North  [3] South  [4] East  [5] West")
        print("Left-Click: Place Point")
        print("[Z]: Undo Last Point")
        print("[Enter]: Finish & Save JSON")
        print("="*40)
        print(f"\n--> Started in Mode: [{state['mode']}]")

        # 5. UI Loop
        while True:
            display_img = img.copy()
            
            # Draw the points and their index numbers
            for pt_mode, pt_list in points.items():
                for i, pt in enumerate(pt_list):
                    cv2.circle(display_img, pt, 4, colors[pt_mode], -1)
                    cv2.putText(display_img, str(i), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[pt_mode], 1)

            cv2.imshow("Create 2D Mesh", display_img)
            key = cv2.waitKey(20) & 0xFF
            
            # Change modes
            if key == ord('1') and state["mode"] != "REF": 
                state["mode"] = "REF"; print(f"--> Mode changed to: [{state['mode']}]")
            elif key == ord('2') and state["mode"] != "N": 
                state["mode"] = "N"; print(f"--> Mode changed to: [{state['mode']}]")
            elif key == ord('3') and state["mode"] != "S": 
                state["mode"] = "S"; print(f"--> Mode changed to: [{state['mode']}]")
            elif key == ord('4') and state["mode"] != "E": 
                state["mode"] = "E"; print(f"--> Mode changed to: [{state['mode']}]")
            elif key == ord('5') and state["mode"] != "W": 
                state["mode"] = "W"; print(f"--> Mode changed to: [{state['mode']}]")
            
            # Undo last point
            elif key == ord('z') or key == ord('Z'):
                current_mode = state["mode"]
                if len(points[current_mode]) > 0:
                    points[current_mode].pop()
                    print(f"Undid last point in [{current_mode}]")
            
            # Finish
            elif key == 13: # Enter key
                if not points["REF"]:
                    print("Error: You must click the Reference (REF) square!")
                    continue
                if len(points["N"]) != len(points["S"]) or len(points["E"]) != len(points["W"]):
                    print(f"Error: Mismatched counts! N:{len(points['N'])}, S:{len(points['S'])}, E:{len(points['E'])}, W:{len(points['W'])}")
                    continue
                if len(points["N"]) == 0 and len(points["E"]) == 0:
                     print("Error: You must click some tick marks!")
                     continue
                break

        cv2.destroyWindow("Create 2D Mesh")

        # 6. Build the 2D Mesh JSON
        # Sort mathematically so it doesn't matter what order you clicked the line
        points["N"].sort(key=lambda pt: pt[0]) 
        points["S"].sort(key=lambda pt: pt[0]) 
        points["E"].sort(key=lambda pt: pt[1]) 
        points["W"].sort(key=lambda pt: pt[1]) 

        print("\nCalculating 2D Mesh intersections...")
        grid_2d = []
        # For every horizontal line (West to East)
        for w_pt, e_pt in zip(points["W"], points["E"]):
            row_points = []
            # Find where it intersects every vertical line (North to South)
            for n_pt, s_pt in zip(points["N"], points["S"]):
                ix, iy = get_line_intersection((n_pt, s_pt), (w_pt, e_pt))
                row_points.append([ix, iy])
            grid_2d.append(row_points)

        ref_pixel = [float(points["REF"][0][0]), float(points["REF"][0][1])]

        final_mapping = {
            "reference_pixel": ref_pixel,
            "grid_2d": grid_2d
        }
        
        # 7. Save to JSON
        json_save_path = os.path.join(os.path.dirname(cam_config.CALIBRATION_REAL_WORLD_PATH), "calibration_map.json")
        
        with open(json_save_path, "w") as f:
            json.dump(final_mapping, f, indent=4)
            
        print("--- 2D Mesh Successfully Computed & Saved! ---")
        print(f"Saved to: {json_save_path}")

    except Exception as e:
        print(f"\nError during calibration: {e}")
        
    finally:
        camera.stop_stream()
        print("Camera stream stopped.")

if __name__ == "__main__":
    main()