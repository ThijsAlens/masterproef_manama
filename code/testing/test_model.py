import torch
from torch.utils.data import DataLoader

import os
import json

from torch_dataset.custom_dataset import CustomDataset
from models_to_test.custom_training.gausian_NNL_loss import GaussianNLLLoss

import testing.config as config

def test_model():
    """
    Test the performance of the trained model on the test dataset and save the results in a JSON file.
    
    Args:
        None
    Returns:
        dict: A dictionary containing the average test loss, predictions, uncertainties, and targets.
    """

    def run_test(model: torch.nn.Module, test_dataset: DataLoader, loss_fn=GaussianNLLLoss()) -> dict[str, any]:
        """
        Test the performance of the trained model on a test dataset.

        Args:
            model (torch.nn.Module): The trained model to test.
            test_dataset (DataLoader): The dataset to evaluate the model on.
            loss_fn (torch.nn.Module): The loss function to compute the test loss (default: GaussianNLLLoss).

        Returns:
            dict: A dictionary containing the average test loss, predictions, uncertainties, and targets.
        """
        device = model.device

        total_test_loss = 0.0
        all_means = []
        all_vars = []
        all_targets = []

        for images, targets in test_dataset:
            mean_pred, var_pred = model(images)

            targets = targets.to(device)
            loss = loss_fn(mean_pred, var_pred, targets)

            total_test_loss += loss.item()

            # Undo normalization done in torch_dataset.custom_dataset to get real-world coordinates in mm
            real_mean_x = (mean_pred[:, 0] * config.OUTPUT_BOUNDS['x_std']) + config.OUTPUT_BOUNDS['x_mean']
            real_mean_y = (mean_pred[:, 1] * config.OUTPUT_BOUNDS['y_std']) + config.OUTPUT_BOUNDS['y_mean']
            real_mean = torch.stack((real_mean_x, real_mean_y), dim=1)
            all_means.append(real_mean.cpu())

            real_var_x = var_pred[:, 0] * (config.OUTPUT_BOUNDS['x_std'] ** 2)
            real_var_y = var_pred[:, 1] * (config.OUTPUT_BOUNDS['y_std'] ** 2)
            real_var = torch.stack((real_var_x, real_var_y), dim=1)
            all_vars.append(real_var.cpu())

            real_target_x = (targets[:, 0] * config.OUTPUT_BOUNDS['x_std']) + config.OUTPUT_BOUNDS['x_mean']
            real_target_y = (targets[:, 1] * config.OUTPUT_BOUNDS['y_std']) + config.OUTPUT_BOUNDS['y_mean']
            real_target = torch.stack((real_target_x, real_target_y), dim=1)
            all_targets.append(real_target.cpu())

        avg_test_loss = total_test_loss / len(test_dataset)

        return {
            "average_loss": avg_test_loss,
            "predictions": torch.cat(all_means),
            "variances": torch.cat(all_vars),
            "targets": torch.cat(all_targets)
        }
    
    test_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_TEST_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    model = torch.load(config.PATH_TO_MODEL)

    results = run_test(model, test_loader)

    for i in range(results["predictions"].shape[0]):
        prediction = results['predictions'][i].numpy()
        variance = results['variances'][i].numpy()
        target = results['targets'][i].numpy()
        
        current_file = json.load(open(config.PATH_TO_RESULTS_FILE, "r"))
        current_file.append({
            "mean_prediction": prediction.tolist(),
            "variance_prediction": variance.tolist(),
            "target": target.tolist()
        })
        with open(config.PATH_TO_RESULTS_FILE, "w") as f:
            json.dump(current_file, f, indent=4)
    
    return results

if __name__ == "__main__":
    test_model()