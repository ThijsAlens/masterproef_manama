"""
This module contains functions to calculate the real world coordinates of the center of the disk using the given image and labeling info.
It does this by:
1. Detecting the lines in the image (using cv2 rectangles) to find the vanishing point.
2. Using the vanishing point and the pixel coordinates of the center to calculate the real world coordinates.
"""

from dataset_creation import config
import cv2
import json
import numpy as np

def find_vanishing_point(lines: list[list[tuple[int, int]]]) -> tuple[float, float]:
    """
    Finds the vanishing point of (at least) two lines. More lines can be used to get a more accurate vanishing point by averaging the intersection points of all pairs of lines.
    
    Args:
        lines (list[list[tuple[int, int]]]): A list of lines represented as lists of tuples of [(x1, y1), (x2, y2)].
        
    Returns:
        tuple[float, float]: The coordinates of the vanishing point as (x, y).
    """
    if len(lines) < 2:
        raise ValueError("At least two lines are required to find a vanishing point.")
    vanishing_points = []
    for i in range(len(lines)-1):
        l1 = lines[i]
        l2 = lines[i+1]

        x1, y1 = l1[0]
        x2, y2 = l1[1]
        x3, y3 = l2[0]
        x4, y4 = l2[1]

        # Convert line to homogeneous coordinates
        p1_h = np.array([x1, y1, 1])
        p2_h = np.array([x2, y2, 1])
        p3_h = np.array([x3, y3, 1])
        p4_h = np.array([x4, y4, 1])

        # A line is the cross product of two points
        line1_h = np.cross(p1_h, p2_h)
        line2_h = np.cross(p3_h, p4_h)

        # The intersection point is the cross product of the two lines
        intersection_h = np.cross(line1_h, line2_h)

        if intersection_h[2] == 0:
            raise ValueError("Lines are parallel, no vanishing point.")

        vanishing_points.append(intersection_h[:2] / intersection_h[2])

    # Return the average of all vanishing points
    return tuple(np.mean(vanishing_points, axis=0))

def calculate_real_world_coordinates(img_path: str, json_path: str) -> None:
    """
    Using the given image (with cv2 rectangles) this function calculates the real world coordinates of the center of the disk.
    
    Args:
        img_path (str): Path to the image file.
        json_path (str): Path to the JSON file containing the labeling info.
        
    Returns:
        None: The function saves the real world coordinates in the same JSON file."""
    
    # 1. Load the image
    img = cv2.imread(img_path)

    # 2. Detect cv2 rectangles in the image to find the lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (config.NUMBER_OF_INNER_CORNERS_X, config.NUMBER_OF_INNER_CORNERS_Y), None)
    if not ret:
        raise ValueError("Could not find the chessboard corners in the image.")
    corners = corners.reshape(-1, 2).tolist()

    vertical_lines = []
    horizontal_lines = []
    for i, corner in enumerate(corners):
        if i < config.NUMBER_OF_INNER_CORNERS_X * (config.NUMBER_OF_INNER_CORNERS_Y - 1):
            vertical_lines.append([(corner[0], corner[1]), (corners[i + config.NUMBER_OF_INNER_CORNERS_X][0], corners[i + config.NUMBER_OF_INNER_CORNERS_X][1])])
        if i % config.NUMBER_OF_INNER_CORNERS_X < config.NUMBER_OF_INNER_CORNERS_X - 1:
            horizontal_lines.append([(corner[0], corner[1]), (corners[i + 1][0], corners[i + 1][1])])

    # 2. Detect lines in the image to find the vanishing point
    vertical_lines_vanishing_point = find_vanishing_point(vertical_lines[:config.NUMBER_OF_INNER_CORNERS_X - 1]) # Use only the first few lines to get a more accurate vanishing point
    new_horizontal_lines = []
    for i in range(len(horizontal_lines)):
        if i % config.NUMBER_OF_INNER_CORNERS_X-1 == 0:
            new_horizontal_lines.append(horizontal_lines[i])
    horizontal_lines_vanishing_point = find_vanishing_point(new_horizontal_lines) # Use only the first few lines to get a more accurate vanishing point

    print(f"vert:")
    for line in vertical_lines[:config.NUMBER_OF_INNER_CORNERS_X - 1]:
        print(line)
    print(f"hor:")
    for line in new_horizontal_lines:
        print(line)
    print(f"Vanishing point for vertical lines: {vertical_lines_vanishing_point}")
    print(f"Vanishing point for horizontal lines: {horizontal_lines_vanishing_point}")

    # 3. Use the vanishing point and pixel coordinates of the center to calculate real world coordinates
    with open(json_path, "r") as f:
        data = json.load(f)
    x = data["pixel"]["x"]
    y = data["pixel"]["y"]

    # project the pixel coordinates onto the axes defined by the vanishing points
    pixel_coords_on_x_axis = find_intersection([(x, y, vertical_lines_vanishing_point[0], vertical_lines_vanishing_point[1]), (vertical_lines[0][0], vertical_lines[0][1], vertical_lines[0][2], vertical_lines[0][3])])
    pixel_coords_on_y_axis = find_intersection([(x, y, horizontal_lines_vanishing_point[0], horizontal_lines_vanishing_point[1]), (horizontal_lines[0][0], horizontal_lines[0][1], horizontal_lines[0][2], horizontal_lines[0][3])])

    print(f"Pixel coordinates on x-axis: {pixel_coords_on_x_axis}")
    print(f"Pixel coordinates on y-axis: {pixel_coords_on_y_axis}")
    print(f"x: {x}, y: {y}")
    # Interpolate to real world coordinates using the known size of the chessboard squares
    for i, corner in enumerate(corners_on_x_axis):
        if corner[0] < pixel_coords_on_x_axis[0]:
            x1, y1 = corner
            n1 = i
        else:
            x2, y2 = corner
            n2 = i
            break
    real_world_x = n1*config.SQUARE_SIZE_MM + ((pixel_coords_on_x_axis[0] - x1) / (x2 - x1)) * config.SQUARE_SIZE_MM

    for i, corner in enumerate(corners_on_y_axis):
        if corner[1] < pixel_coords_on_y_axis[1]:
            x1, y1 = corner
            n1 = i
        else:
            x2, y2 = corner
            n2 = i
            break

    real_world_y = n1*config.SQUARE_SIZE_MM + ((pixel_coords_on_y_axis[1] - y1) / (y2 - y1)) * config.SQUARE_SIZE_MM

    # 4. Save the real world coordinates back to the JSON file
    data["real_world"] = {
        "x": real_world_x,
        "y": real_world_y
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    

