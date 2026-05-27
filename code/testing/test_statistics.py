import json
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

import testing.config as config

def generate_fancy_error_uncertainty_graph(json_file_path: str, dir_to_save: str, show: bool=False) -> None:
    """
    Thanks gemini :)
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    errors = []
    uncertainties = []
    
    for entry in data:
        pred = np.array(entry["mean_prediction"])
        target = np.array(entry["target"])
        uncertainty = np.array(np.sqrt(entry["total_variance_predictions"]))
        
        error = np.abs(np.subtract(pred, target))
        
        errors.extend(error)
        uncertainties.extend(uncertainty)

    max_x = max(uncertainties) * 1.05

    # --- 2. Create Two Stacked Plots ---
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(hspace=0.05) 

    # --- 3. Define the Mathematical Bounds (1-Sigma, 2-Sigma, 3-Sigma) ---
    # Assuming your image uses standard y = x, y = 2x, y = 3x bounds
    x_vals = np.linspace(0, max_x, 100)
    sigma_1 = x_vals * 1
    sigma_2 = x_vals * 2
    sigma_3 = x_vals * 3

    # --- 4. Loop through BOTH axes to draw the data, lines, and shading ---
    for ax in (ax_top, ax_bottom):
        # The Shading (fill_between)
        ax.fill_between(x_vals, 0, sigma_1, color='lightblue', alpha=0.3)
        ax.fill_between(x_vals, sigma_1, sigma_2, color='lightgreen', alpha=0.3)
        ax.fill_between(x_vals, sigma_2, sigma_3, color='bisque', alpha=0.5)
        # The top zone needs an arbitrarily high upper limit (e.g., 1000) so it doesn't get cut off
        ax.fill_between(x_vals, sigma_3, 1000, color='lightpink', alpha=0.4) 

        # The Dashed Lines
        ax.plot(x_vals, sigma_1, color='royalblue', linestyle='--', alpha=0.8)
        ax.plot(x_vals, sigma_2, color='forestgreen', linestyle='--', alpha=0.8)
        ax.plot(x_vals, sigma_3, color='mediumvioletred', linestyle='--', alpha=0.8)

        # The Scatter Plot
        ax.scatter(uncertainties, errors, alpha=0.7, edgecolors='white', color='tab:blue', zorder=5)

    # --- 5. Set the Y-Limits to "Break" the space ---
    ax_top.set_ylim(250, 600)   # Outlier zone
    ax_bottom.set_ylim(0, 100)  # Main cluster zone
    ax_bottom.set_xlim(0, max_x)

    # --- 6. Hide the spines to merge them visually ---
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False) 

    # --- 7. Draw the slanted // lines manually ---
    d = .015  
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  

    kwargs.update(transform=ax_bottom.transAxes)  
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

    # --- 8. Labels ---
    ax_bottom.set_xlabel('Uncertainty of Predictions (Sigma in mm)')
    fig.text(0.04, 0.5, 'Absolute Error (mm)', va='center', rotation='vertical')

    plt.savefig(f"{dir_to_save}/fancy_error_uncertainty_graph_{json_file_path[json_file_path.rindex('/') + 1:][:-5]}.png")
    plt.show() if show else plt.close()


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
        uncertainty = np.array(np.sqrt(entry["total_variance_predictions"]))
        
        error = np.abs(np.subtract(pred, target))
        
        errors.extend(error)
        uncertainties.extend(uncertainty)
        
    plt.figure(figsize=(10, 6))
    
    # 1. Establish strict limits based strictly on the maximum data points
    x_limit = max(uncertainties) * 1.05
    y_limit = max(errors) * 1.05
    x_vals = np.linspace(0, x_limit, 100) 

    # 2. Use np.minimum to cap the zone boundaries at y_limit
    line1 = np.minimum(x_vals, y_limit)
    line2 = np.minimum(2 * x_vals, y_limit)
    line3 = np.minimum(3 * x_vals, y_limit)

    # 3. Fill the zones using the capped boundaries
    plt.fill_between(x_vals, 0, line1, color='#add8e6', alpha=0.3, label='Cautious (Underconfident)')
    plt.fill_between(x_vals, line1, line2, color='#90ee90', alpha=0.3, label='Sweet Spot (Perfect Calibration)')
    plt.fill_between(x_vals, line2, line3, color='#ffcc99', alpha=0.4, label='Warning (Overconfident)')
    plt.fill_between(x_vals, line3, y_limit, color='#ff9999', alpha=0.3, label='Danger (Blindly Confident)')

    # 4. Plot the dashed boundary lines (truncated at y_limit for visual cleanliness)
    plt.plot(x_vals[x_vals <= y_limit], x_vals[x_vals <= y_limit], color='blue', linestyle='--', alpha=0.7, zorder=3)
    plt.plot(x_vals[2 * x_vals <= y_limit], (2 * x_vals)[2 * x_vals <= y_limit], color='green', linestyle='--', alpha=0.7, zorder=3)
    plt.plot(x_vals[3 * x_vals <= y_limit], (3 * x_vals)[3 * x_vals <= y_limit], color='purple', linestyle='--', alpha=0.7, zorder=3)

    # 5. Plot the data points
    plt.scatter(uncertainties, errors, alpha=0.7, edgecolors='white', zorder=4)

    # 6. Force the plot to lock exactly to the data boundaries
    plt.xlim(0, x_limit)
    plt.ylim(0, y_limit)
    
    plt.xlabel("Uncertainty of Predictions (Sigma in mm)")
    plt.ylabel("Absolute Error (mm)")

    plt.savefig(f"{dir_to_save}/error_uncertainty_graph_{json_file_path[json_file_path.rindex('/') + 1:][:-5]}.png")
    plt.show() if show else plt.close()

    return

def binary_scoring(json_file_path: str, confidence: float, acceptable_error_mm: float) -> None:
    """
    Evaluates the model's performance using a binary scoring system based on the confidence interval and acceptable error threshold.
    
    Args:
        json_file_path (str): The path to the JSON file containing the model results.
        confidence (float): The confidence level to determine if a prediction is considered "confident".
        acceptable_error_mm (float): The maximum acceptable error in millimeters for a prediction to be considered "accurate".
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Until what error threshold (in mm) does the model's confidence interval cover the true target?
    chi2_val = scipy.stats.chi2.ppf(confidence, df=2)
    max_allowed_variance = (acceptable_error_mm ** 2) / chi2_val

    confident_in_success = np.zeros(len(data), dtype=bool)
    actual_success = np.zeros(len(data), dtype=bool)

    for i, entry in enumerate(data):
        confident_in_success[i] = True if np.max(entry["total_variance_predictions"]) <= max_allowed_variance else False
        error = np.abs(np.subtract(entry["mean_prediction"], entry["target"]))
        actual_success[i] = math.sqrt(np.sum(error ** 2)) <= acceptable_error_mm

    print(f"max_allowed_variance: {max_allowed_variance:.2f} mm^2 | max_allowed_sigma: {math.sqrt(max_allowed_variance):.2f} mm")
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
    binary_scoring(config.JSON_FILE_PATH, config.CONFIDENCE, config.ACCEPTABLE_ERROR_THRESHOLD_MM)
    generate_error_uncertainty_graph(config.JSON_FILE_PATH, config.DIR_TO_SAVE, show=config.SHOW_GRAPH)
    generate_fancy_error_uncertainty_graph(config.JSON_FILE_PATH, config.DIR_TO_SAVE, show=config.SHOW_GRAPH)