import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

import testing.config as config



def generate_error_variance_graph(json_file_path: str, dir_to_save: str, show: bool=False) -> None:
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
        pred = np.array(entry["prediction"])
        target = np.array(entry["target"])
        uncertainty = np.array(entry["uncertainty"])
        
        error = np.abs(np.subtract(pred, target))
        
        errors.append(error)
        uncertainties.append(uncertainty)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(errors, uncertainties, alpha=0.5)
    plt.axline((0, 0), slope=1, color='red', linestyle='--')
    plt.xlabel("Error")
    plt.ylabel("Uncertainty of Predictions")
    plt.title("Error-uncertainty Graph")
    plt.grid()
    plt.savefig(f"{dir_to_save}/error_variance_graph_{json_file_path[json_file_path.rindex('/') + 1:][:-5]}.png")
    plt.show() if show else plt.close()

    return

def calculate_score(json_file_path: str, _lambda: float) -> None:
    """
    Calculates the score for each entry in the JSON file using the formula:
    score = exp(1 - ( (y - mean)^2 / var^2 ) - log( var^2 / (y - mean)^2 ))
    
    Args:
        json_file_path (str): The path to the JSON file containing the model results.
        _lambda (float): The lambda parameter for the score calculation.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    for entry in data:
        pred = np.array(entry["prediction"])
        target = np.array(entry["target"])
        uncertainty = np.array(entry["uncertainty"])
        
        error = np.abs(np.subtract(pred, target))
        
        score = np.exp(_lambda * (1 - ( (np.square(error)) / (np.square(uncertainty)) ) - np.log( np.square(uncertainty) / (np.square(error)) )) )
        combined_score = math.prod(score)
        print(f"Prediction: {pred}, Target: {target}, Uncertainty: {uncertainty}, Score: {score}\nCombined Score: {combined_score}")

if __name__ == "__main__":
    os.makedirs(config.DIR_TO_SAVE, exist_ok=True)
    generate_error_variance_graph(config.JSON_FILE_PATH, config.DIR_TO_SAVE, show=config.SHOW_GRAPH)
    calculate_score(config.JSON_FILE_PATH, config.LAMBDA)