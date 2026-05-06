# Script that outputs the bounds of the dataset, which can be used for normalization in the CustomDataset class. Run this script once to get the bounds, and then use those bounds in the CustomDataset class to avoid having to calculate them every time.

import json
import os
import numpy as np

if __name__ == "__main__":
    data_dir = "data/regression_use"
    bounds = {
        "pixel": {
            "x_mean": None,
            "x_std": None,
            "y_mean": None,
            "y_std": None
        },
        "world": {
            "x_mean": None,
            "x_std": None,
            "y_mean": None,
            "y_std": None
        }
    }
    
    all_x_p, all_y_p = [], []
    all_x_w, all_y_w = [], []
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(subdir_path, file)
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            x_pixel = json_data["pixel"]["x"]
            y_pixel = json_data["pixel"]["y"]
            x_world = json_data["world"]["x"]
            y_world = json_data["world"]["y"]

            all_x_p.append(x_pixel)
            all_y_p.append(y_pixel)
            all_x_w.append(x_world)
            all_y_w.append(y_world)
        
        bounds["pixel"]["x_mean"] = np.mean(all_x_p)
        bounds["pixel"]["x_std"] = np.std(all_x_p)
        bounds["pixel"]["y_mean"] = np.mean(all_y_p)
        bounds["pixel"]["y_std"] = np.std(all_y_p)
        bounds["world"]["x_mean"] = np.mean(all_x_w)
        bounds["world"]["x_std"] = np.std(all_x_w)
        bounds["world"]["y_mean"] = np.mean(all_y_w)
        bounds["world"]["y_std"] = np.std(all_y_w)

    print(f"Bounds for pixel coordinates:\n{json.dumps(bounds['pixel'], indent=4)}")
    print(f"Bounds for world coordinates:\n{json.dumps(bounds['world'], indent=4)}")
    with open("dataset/dataset_bounds.json", 'w') as f:
        json.dump(bounds, f, indent=4)
