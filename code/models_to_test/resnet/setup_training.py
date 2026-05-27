import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import copy
import os
import json

from models_to_test.resnet.resnet import DeepEnsemble, ResNetRegressionModel
from models_to_test.custom_training.gausian_NNL_loss import GaussianNLLLoss
import models_to_test.resnet.config as config
from torch_dataset.custom_dataset import CustomDataset 

def set_seed(seed: int = 1) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to set for random number generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def training_step(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Perform a single training step for the given model, including adversarial training.
    
    Args:
        model (torch.nn.Module): The model to train.
        images (torch.Tensor): The input images for the training step.
        targets (torch.Tensor): The target values for the training step.
        loss_fn (torch.nn.Module): The loss function to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
        
    Returns:
        float: The computed training loss for this step.
    """
    model.train()
    images.requires_grad = True
    
    # Forward pass
    mean_pred, var_pred = model(images)
    loss_orig = loss_fn(mean_pred, var_pred, targets)
    
    # Calculate adversarial direction
    model.zero_grad()
    loss_orig.backward(retain_graph=True) 
    
    # Generate Adversarial images and do second forward pass
    epsilon = 0.01 * (images.max() - images.min()) 
    with torch.no_grad():
        images_adv = images + epsilon * images.grad.sign()
        
    mean_adv, var_adv = model(images_adv)
    loss_adv = loss_fn(mean_adv, var_adv, targets)
    
    # Combine losses and step optimizer
    total_loss = loss_orig + loss_adv
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def validation_step(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module) -> float:
    """
    Perform a single validation step for the given model.
    
    Args:
        model (torch.nn.Module): The model to validate.
        images (torch.Tensor): The input images for the validation step.
        targets (torch.Tensor): The target values for the validation step.
        loss_fn (torch.nn.Module): The loss function to compute the validation loss.

    Returns:
        float: The computed validation loss for this step.
    """
    model.eval()
    with torch.no_grad():
        mean_pred, var_pred = model(images)
        val_loss = loss_fn(mean_pred, var_pred, targets)
    return val_loss.item()

def train_ensemble_bagging(num_models: int=5, training_dataset: DataLoader=None, validation_dataset: DataLoader=None, model_parameters: dict[str, any]=None, epochs: int=10, lr:int=1e-3, batch_size: int=32, bagging_ratio: float=None) -> DeepEnsemble:
    """
    Train a deep ensemble using bagging (bootstrap aggregating).
    
    Args:
        num_models (int): The number of models to include in the ensemble.
        training_dataset (DataLoader): The dataset to train on.
        validation_dataset (DataLoader): The dataset to validate on.
        model_parameters (dict[str, any]): A dictionary of parameters to initialize each model in the ensemble.
        epochs (int): The number of training epochs for each model.
        lr (float): The learning rate for training each model.
        batch_size (int): The batch size for training and validation.
        bagging_ratio (float): The ratio of the dataset to sample for each model (between 0 and 1). If None, the entire dataset is used for each model (no bagging).
    
    Returns:
        DeepEnsemble: A trained deep ensemble model.
    """
    if num_models <= 0:
        raise ValueError("num_models must be a positive integer.")
    if training_dataset is None or validation_dataset is None:
        raise ValueError("training_dataset and validation_dataset must be provided to train the models in the ensemble.")
    if model_parameters is None:
        raise ValueError("model_parameters must be provided to initialize the models in the ensemble.")
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if lr <= 0:
        raise ValueError("lr must be a positive float.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if bagging_ratio is not None and (bagging_ratio <= 0 or bagging_ratio > 1):
        raise ValueError("bagging_ratio must be between 0 and 1 or None.")
    
    ensemble_models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    for i in range(num_models):
        # Create a unique TensorBoard run name for this ensemble member
        run_name = f"resnet_model_{i}_{timestamp}"
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        
        # Initialize model, optimizer, scheduler, and loss function
        model = ResNetRegressionModel(**model_parameters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        loss_fn = GaussianNLLLoss()
        
        # Bagging
        if bagging_ratio is None:
            indices = list(range(len(training_dataset)))
        else:
            bag_size = int(len(training_dataset) * bagging_ratio)
            indices = torch.randint(0, len(training_dataset), (bag_size,)).tolist()
        train_loader = DataLoader(Subset(training_dataset, indices), batch_size=batch_size, shuffle=True)
        
        # Variables to keep track of the best model based on validation loss
        best_avg_val_loss = float('inf')
        best_model_wts = None

        print(f"Training Model {i}...")
        for epoch in tqdm(range(epochs)):
            # Train
            total_train_loss = 0.0
            num_train_batches = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                batch_training_loss = training_step(model, images, targets, loss_fn, optimizer)
                total_train_loss += batch_training_loss
                num_train_batches += 1
            avg_train_loss = total_train_loss / num_train_batches
            writer.add_scalar(f"NLL_Loss/Train_Epoch", avg_train_loss, epoch)

            # Validate
            total_val_loss = 0.0
            num_val_batches = 0
            for images, targets in validation_dataset:
                images, targets = images.to(device), targets.to(device)
                batch_val_loss = validation_step(model, images, targets, loss_fn)
                total_val_loss += batch_val_loss
                num_val_batches += 1
            avg_val_loss = total_val_loss / num_val_batches
            writer.add_scalar(f"NLL_Loss/Validation", avg_val_loss, epoch)
            scheduler.step(avg_val_loss)

            # Keep track of the best model based on validation loss
            if avg_val_loss < best_avg_val_loss:
                best_avg_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # Save the best model for this ensemble member
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            print(f"Model {i} - validation loss: {best_avg_val_loss:.4f}")
            os.makedirs(config.PATH_TO_SAVE_MODEL_DIR, exist_ok=True)
            save_path = os.path.join(config.PATH_TO_SAVE_MODEL_DIR, f"{run_name}.pth")
            torch.save(model.state_dict(), save_path)

        ensemble_models.append(model)
        writer.close()

    # Combine the trained models into a deep ensemble and save it
    ensemble = DeepEnsemble(ensemble_models)
    torch.save(ensemble.state_dict(), os.path.join(config.PATH_TO_SAVE_MODEL_DIR, f"resnet_ensemble_{datetime.now().strftime('%Y-%m-%d')}.pth"))

    return ensemble

def test_ensemble(ensemble: DeepEnsemble, test_dataset: DataLoader, loss_fn=GaussianNLLLoss()) -> dict[str, any]:
    """
    Test the performance of the trained ensemble on a test dataset.

    Args:
        ensemble (DeepEnsemble): The trained ensemble model to test.
        test_dataset (DataLoader): The dataset to evaluate the ensemble on.
        loss_fn (torch.nn.Module): The loss function to compute the test loss (default: GaussianNLLLoss).

    Returns:
        dict: A dictionary containing the average test loss, predictions, uncertainties, and targets.
    """
    device = ensemble.device
    
    total_test_loss = 0.0
    all_means = []
    all_vars = []
    all_targets = []

    for images, targets in test_dataset:
        mean_pred, var_pred = ensemble.predict(images)
        
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

def save_results(results: dict, save_path: str, print_results: bool = True) -> None:
    """
    Save the test results to a JSON file.

    Args:
        results (dict): The dictionary containing the test results to save.
        save_path (str): The file path where the results should be saved.
        print_results (bool): Whether to print the results to the console (default: True).

    Returns:
        None
    """
    with open(save_path, "w") as f:
        json.dump([], f)  # Clear the file before appending results

    for i in range(results['predictions'].shape[0]):
        prediction = results['predictions'][i].numpy()
        variance = results['variances'][i].numpy()
        target = results['targets'][i].numpy()

        if print_results:
            print(f"Sample {i:<3} | Pred: [{prediction[0]:>5.1f}, {prediction[1]:>5.1f}] | Var: [{variance[0]:>5.1f}, {variance[1]:>5.1f}] mm | Target: [{target[0]:>5.1f}, {target[1]:>5.1f}]")

        current_file = json.load(open(save_path, "r"))
        current_file.append({
            "mean_prediction": prediction.tolist(),
            "variance_prediction": variance.tolist(),
            "target": target.tolist()
        })
        with open(save_path, "w") as f:
            json.dump(current_file, f, indent=4)
    return

if __name__ == "__main__":
    set_seed(config.SEED)
    os.makedirs(config.PATH_TO_RESULTS_DIR, exist_ok=True)

    train_loader = CustomDataset(source_dir=config.PATH_TO_TRAIN_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS)
    val_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_VALIDATION_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    
    deep_ensemble = train_ensemble_bagging(num_models=config.NUM_MODELS, training_dataset=train_loader, validation_dataset=val_loader, model_parameters=config.MODEL_PARAMETERS, epochs=config.EPOCHS, lr=config.LR, batch_size=config.BATCH_SIZE, bagging_ratio=config.BAGGING_SAMPLE_RATIO)


    val_save_path = os.path.join(config.PATH_TO_RESULTS_DIR, f"validationset_results_{datetime.now().strftime('%Y-%m-%d')}.json")
    val_results = test_ensemble(deep_ensemble, val_loader)
    print(f"Average Validation Loss: {val_results['average_loss']:.4f}")
    save_results(val_results, val_save_path)