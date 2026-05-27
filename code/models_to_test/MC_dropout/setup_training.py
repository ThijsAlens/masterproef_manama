import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


from torchvision import datasets, transforms

import random
import numpy as np
import copy
import os
from tqdm import tqdm
import json

import models_to_test.MC_dropout.config as config
from models_to_test.MC_dropout.mc_dropout import MC_Dropout, SimpleCNNRegressionModelDropout
from torch_dataset.custom_dataset import CustomDataset
from models_to_test.custom_training.gausian_NNL_loss import GaussianNLLLoss

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
    A single training step for a model, including forward pass, loss computation, backward pass, and optimizer step.

    Args:
        model (torch.nn.Module): The model to train.
        images (torch.Tensor): The input images for the training step.
        targets (torch.Tensor): The target labels for the training step.
        loss_fn (torch.nn.Module): The loss function to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.

    Returns:
        float: The computed training loss for the current step.
    """
    model.train()
    images.requires_grad = True
    
    # Forward pass
    mean_pred, var_pred = model(images)
    loss_orig = loss_fn(mean_pred, var_pred, targets)

    # Calculate adversarial direction
    model.zero_grad()
    loss_orig.backward(retain_graph=True) 
    
    # Generate Adversarial images
    epsilon = 0.01 * (images.max() - images.min()) 
    with torch.no_grad():
        images_adv = images + epsilon * images.grad.sign()
        
    # Forward pass with adversarial images
    mean_adv, var_adv = model(images_adv)
    loss_adv = loss_fn(mean_adv, var_adv, targets)

    total_loss = loss_orig + loss_adv
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def validation_step(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module) -> float:
    """
    A single validation step for a model, including forward pass and loss computation.

    Args:
        model (torch.nn.Module): The model to validate.
        images (torch.Tensor): The input images for the validation step.
        targets (torch.Tensor): The target labels for the validation step.
        loss_fn (torch.nn.Module): The loss function to compute the validation loss.

    Returns:
        float: The computed validation loss for the current step.
    """
    model.eval()
    with torch.no_grad():
        mean_pred, var_pred = model(images)
        val_loss = loss_fn(mean_pred, var_pred, targets)
    return val_loss.item()

def train_mc_dropout(train_loader: DataLoader=None, val_loader: DataLoader=None, model_parameters: dict=None, epochs: int=10, lr: float=1e-3) -> None:
    """
    Train a model using Monte Carlo Dropout.

    Args:
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        model_parameters (dict): The parameters for initializing the model.
        epochs (int): The number of training epochs for each model (default: 10).
        lr (float): The learning rate for the optimizer (default: 1e-3).

    Returns:
        None
    """
    if train_loader is None or val_loader is None:
        raise ValueError("train_loader and val_loader must be provided to train the model.")
    if model_parameters is None:
        raise ValueError("model_parameters must be provided to initialize the models in the ensemble.")
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if lr <= 0:
        raise ValueError("lr must be a positive float.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Initialize TensorBoard writer
    run_name = f"MC_Dropout_{timestamp}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # Initialize model, optimizer, scheduler, and loss function
    model = SimpleCNNRegressionModelDropout(**model_parameters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    loss_fn = GaussianNLLLoss()

    # Variables to keep track of the best model based on validation loss
    best_avg_val_loss = float('inf')
    best_model_wts = None

    print(f"Training MC Dropout model...")
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
        for images, targets in val_loader:
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
    
    # Save the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"Best validation loss: {best_avg_val_loss:.4f}")

        mc_dropout_model = MC_Dropout(model, n_samples=config.NUM_SAMPLES)
        os.makedirs(config.PATH_TO_SAVE_MODEL_DIR, exist_ok=True)
        save_path = os.path.join(config.PATH_TO_SAVE_MODEL_DIR, f"MC_dropout_{datetime.now().strftime('%Y-%m-%d')}.pth")
        torch.save(mc_dropout_model.state_dict(), save_path)
    return mc_dropout_model

def test_mc_dropout(mc_dropout: MC_Dropout, test_loader: DataLoader, loss_fn: torch.nn.Module=GaussianNLLLoss()) -> None:
    """
    Test a model using Monte Carlo Dropout.

    Args:
        mc_dropout (MC_Dropout): The Monte Carlo Dropout model to test.
        test_loader (DataLoader): The data loader for the test dataset.
        loss_fn (torch.nn.Module): The loss function to compute the test loss.

    Returns:
        None
    """
    device = mc_dropout.device

    total_test_loss = 0.0
    all_means = []
    all_vars = []
    all_targets = []

    for images, targets in test_loader:
        mean_pred, var_pred = mc_dropout.predict(images)

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

    avg_test_loss = total_test_loss / len(test_loader)

    return {
        "average_test_loss": avg_test_loss,
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

    train_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_TRAIN_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_VALIDATION_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    
    model = train_mc_dropout(train_loader, val_loader, model_parameters=config.MODEL_PARAMETERS, epochs=config.EPOCHS, lr=config.LR)


    val_save_path = os.path.join(config.PATH_TO_RESULTS_DIR, f"validationset_results_{datetime.now().strftime('%Y-%m-%d')}.json")
    val_results = test_mc_dropout(model, val_loader)
    print(f"Average validation loss: {val_results['average_test_loss']:.4f}")
    save_results(val_results, val_save_path)