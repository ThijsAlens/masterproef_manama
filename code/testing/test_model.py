import torch
from torch.utils.data import DataLoader

import json
import time

from torch_dataset.custom_dataset import CustomDataset
from models_to_test.custom_training.gausian_NNL_loss import GaussianNLLLoss

import testing.config as config

def set_seed(seed: int = 1) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to set for random number generators.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

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
        all_aleatoric_vars = []
        all_epistemic_vars = []
        all_total_vars = []
        all_targets = []

        avg_inference_time_per_batch = 0.0

        for images, targets in test_dataset:
            images = images.to(device)

            start_time = time.time()
            mean_pred, (total_var, alea_var, epis_var) = model.predict(images, split_variances=True)
            end_time = time.time()

            avg_inference_time_per_batch += end_time - start_time

            targets = targets.to(device)
            loss = loss_fn(mean_pred, total_var, targets)

            total_test_loss += loss.item()

            # Undo normalization done in torch_dataset.custom_dataset to get real-world coordinates in mm
            real_mean_x = (mean_pred[:, 0] * config.OUTPUT_BOUNDS['x_std']) + config.OUTPUT_BOUNDS['x_mean']
            real_mean_y = (mean_pred[:, 1] * config.OUTPUT_BOUNDS['y_std']) + config.OUTPUT_BOUNDS['y_mean']
            real_mean = torch.stack((real_mean_x, real_mean_y), dim=1)
            all_means.append(real_mean.cpu())

            real_aleatoric_var_x = alea_var[:, 0] * (config.OUTPUT_BOUNDS['x_std'] ** 2)
            real_aleatoric_var_y = alea_var[:, 1] * (config.OUTPUT_BOUNDS['y_std'] ** 2)
            real_aleatoric_var = torch.stack((real_aleatoric_var_x, real_aleatoric_var_y), dim=1)
            all_aleatoric_vars.append(real_aleatoric_var.cpu())

            real_epistemic_var_x = epis_var[:, 0] * (config.OUTPUT_BOUNDS['x_std'] ** 2)
            real_epistemic_var_y = epis_var[:, 1] * (config.OUTPUT_BOUNDS['y_std'] ** 2)
            real_epistemic_var = torch.stack((real_epistemic_var_x, real_epistemic_var_y), dim=1)
            all_epistemic_vars.append(real_epistemic_var.cpu())

            real_total_var_x = total_var[:, 0] * (config.OUTPUT_BOUNDS['x_std'] ** 2)
            real_total_var_y = total_var[:, 1] * (config.OUTPUT_BOUNDS['y_std'] ** 2)
            real_total_var = torch.stack((real_total_var_x, real_total_var_y), dim=1)
            all_total_vars.append(real_total_var.cpu())

            real_target_x = (targets[:, 0] * config.OUTPUT_BOUNDS['x_std']) + config.OUTPUT_BOUNDS['x_mean']
            real_target_y = (targets[:, 1] * config.OUTPUT_BOUNDS['y_std']) + config.OUTPUT_BOUNDS['y_mean']
            real_target = torch.stack((real_target_x, real_target_y), dim=1)
            all_targets.append(real_target.cpu())

        avg_test_loss = total_test_loss / len(test_dataset)
        avg_aleatoric_var = torch.cat(all_aleatoric_vars).mean().item()
        avg_epistemic_var = torch.cat(all_epistemic_vars).mean().item()
        avg_inference_time_per_batch /= len(test_dataset)
        print(f"Average Test Loss: {avg_test_loss:.2f}")
        print(f"Average Aleatoric Variance: {avg_aleatoric_var:.2f}")
        print(f"Average Epistemic Variance: {avg_epistemic_var:.2f}")

        return {
            "average_loss": avg_test_loss,
            "mean_predictions": torch.cat(all_means),
            "aleatoric_variances_predictions": torch.cat(all_aleatoric_vars),
            "epistemic_variances_predictions": torch.cat(all_epistemic_vars),
            "total_variances_predictions": torch.cat(all_total_vars),
            "targets": torch.cat(all_targets),
            "average_inference_time_per_batch": avg_inference_time_per_batch
        }
    
    test_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_TEST_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    model = torch.load(config.PATH_TO_MODEL, weights_only=False)

    results = run_test(model, test_loader)

    print(f"Average inference time per image: {results['average_inference_time_per_batch']*1000 / config.BATCH_SIZE:.4f} ms")

    with open(config.PATH_TO_RESULTS_FILE, "w") as f:
        json.dump([], f, indent=4)

    for i in range(results["mean_predictions"].shape[0]):
        prediction = results['mean_predictions'][i].numpy()
        aleatoric_variance = results['aleatoric_variances_predictions'][i].numpy()
        epistemic_variance = results['epistemic_variances_predictions'][i].numpy()
        total_variance = results['total_variances_predictions'][i].numpy()
        target = results['targets'][i].numpy()
        
        current_file = json.load(open(config.PATH_TO_RESULTS_FILE, "r"))
        current_file.append({
            "mean_prediction": prediction.tolist(),
            "aleatoric_variance_predictions": aleatoric_variance.tolist(),
            "epistemic_variance_predictions": epistemic_variance.tolist(),
            "total_variance_predictions": total_variance.tolist(),
            "target": target.tolist()
        })
        with open(config.PATH_TO_RESULTS_FILE, "w") as f:
            json.dump(current_file, f, indent=4)
    
    return results

if __name__ == "__main__":
    set_seed()
    test_model()