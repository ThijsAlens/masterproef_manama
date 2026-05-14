import json
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

import testing.config as config



def generate_error_uncertainty_graph(json_file_path: str, dir_to_save: str, show: bool=False) -> None:
    """
    Generates a graph that shows the relationship between the error and the variance of the predictions.
    
    Args:
        json_file_path (str): The path to the JSON file containing the model results.
        dir_to_save (str): The directory where the graph will be saved.
        show (bool): Whether to display the graph after saving it (default: False).
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    errors = []
    uncertainties = []
    
    for entry in data:
        pred = np.array(entry["mean_prediction"])
        target = np.array(entry["target"])
        uncertainty = np.array(np.sqrt(entry["variance_prediction"]))
        
        error = np.abs(np.subtract(pred, target))
        
        errors.extend(error)
        uncertainties.extend(uncertainty)
        
    plt.figure(figsize=(10, 6))
    
    x_limit = max(uncertainties) * 1.05
    y_limit = max(errors) * 1.05
    x_vals = np.linspace(0, x_limit, 100) 
    plt.fill_between(x_vals, 0, x_vals, color='#add8e6', alpha=0.3, label='Cautious (Underconfident)')
    plt.fill_between(x_vals, x_vals, 2 * x_vals, color='#90ee90', alpha=0.3, label='Sweet Spot (Perfect Calibration)')
    plt.fill_between(x_vals, 2 * x_vals, 3 * x_vals, color='#ffcc99', alpha=0.4, label='Warning (Overconfident)')
    plt.fill_between(x_vals, 3 * x_vals, y_limit, color='#ff9999', alpha=0.3, label='Danger (Blindly Confident)')

    plt.plot(x_vals, x_vals, color='blue', linestyle='--', alpha=0.7, zorder=3)
    plt.plot(x_vals, 2 * x_vals, color='green', linestyle='--', alpha=0.7, zorder=3)
    plt.plot(x_vals, 3 * x_vals, color='purple', linestyle='--', alpha=0.7, zorder=3)

    plt.scatter(uncertainties, errors, alpha=0.7, edgecolors='white', zorder=4)

    # plt.xlim(0, 25)
    # plt.ylim(0, 25)
    plt.xlabel("Uncertainty of Predictions (Sigma in mm)")
    plt.ylabel("Absolute Error (mm)")
    plt.title(f"Error-Uncertainty Graph ({json_file_path[json_file_path.rindex('/') + 1:][:-5]})")

    plt.savefig(f"{dir_to_save}/error_uncertainty_graph_{json_file_path[json_file_path.rindex('/') + 1:][:-5]}.png")
    plt.show() if show else plt.close()

    return

def binary_scoring(json_file_path: str, confidence_interval: float, acceptable_error_mm: float) -> None:
    """
    Evaluates the model's performance using a binary scoring system based on the confidence interval and acceptable error threshold.
    
    Args:
        json_file_path (str): The path to the JSON file containing the model results.
        confidence_interval (float): The confidence interval to determine if a prediction is considered "confident".
        acceptable_error_mm (float): The maximum acceptable error in millimeters for a prediction to be considered "accurate".
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Until what error threshold (in mm) does the model's confidence interval cover the true target?
    chi2_val = scipy.stats.chi2.ppf(confidence_interval, df=2)
    max_allowable_variance = (acceptable_error_mm ** 2) / chi2_val

    confident_in_success = np.zeros(len(data), dtype=bool)
    actual_success = np.zeros(len(data), dtype=bool)

    for i, entry in enumerate(data):
        confident_in_success[i] = True if np.max(entry["variance_prediction"]) <= max_allowable_variance else False
        error = np.abs(np.subtract(entry["mean_prediction"], entry["target"]))
        actual_success[i] = math.sqrt(np.sum(error ** 2)) <= acceptable_error_mm

    print(f"max_allowable_variance: {max_allowable_variance:.2f} mm^2 | max_allowable_sigma: {math.sqrt(max_allowable_variance):.2f} mm")
    TP = (confident_in_success & actual_success).sum()
    FP = (confident_in_success & ~actual_success).sum()
    FN = (~confident_in_success & actual_success).sum()
    TN = (~confident_in_success & ~actual_success).sum()
    
    print(f"True positives (Confident and Accurate): {TP}")
    print(f"False positives (Confident but Inaccurate): {FP}")
    print(f"False negatives (Unconfident but Accurate): {FN}")
    print(f"True negatives (Unconfident and Inaccurate): {TN}")

    return


if __name__ == "__main__":
    binary_scoring(config.JSON_FILE_PATH, config.CONFIDENCE_INTERVAL, config.ACCEPTABLE_ERROR_THRESHOLD_MM)
    generate_error_uncertainty_graph(config.JSON_FILE_PATH, config.DIR_TO_SAVE, show=config.SHOW_GRAPH)