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

def training_step(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, writer: SummaryWriter, epoch: int) -> float:
    """
    A single training step for a model, including forward pass, loss computation, backward pass, and optimizer step.

    Args:
        model (torch.nn.Module): The model to train.
        images (torch.Tensor): The input images for the training step.
        targets (torch.Tensor): The target labels for the training step.
        loss_fn (torch.nn.Module): The loss function to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer to log training metrics.
        epoch (int): The current epoch number for logging purposes.

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
    
    writer.add_scalar(f"{loss_fn.__class__.__name__}/Train", total_loss.item(), epoch)
    return total_loss.item()

def validation_step(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module, writer: SummaryWriter, epoch: int) -> float:
    """
    A single validation step for a model, including forward pass and loss computation.

    Args:
        model (torch.nn.Module): The model to validate.
        images (torch.Tensor): The input images for the validation step.
        targets (torch.Tensor): The target labels for the validation step.
        loss_fn (torch.nn.Module): The loss function to compute the validation loss.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer to log validation metrics.
        epoch (int): The current epoch number for logging purposes.

    Returns:
        float: The computed validation loss for the current step.
    """
    model.eval()
    with torch.no_grad():
        mean_pred, var_pred = model(images)
        val_loss = loss_fn(mean_pred, var_pred, targets)
        writer.add_scalar(f"{loss_fn.__class__.__name__}/Val", val_loss.item(), epoch)
    return val_loss.item()

def train_mc_dropout(train_loader: DataLoader, val_loader: DataLoader, model_parameters: dict, num_epochs: int=10, lr: float=1e-3) -> None:
    """
    Train a model using Monte Carlo Dropout.

    Args:
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        model_parameters (dict): The parameters for initializing the model.
        num_epochs (int): The number of epochs to train the model (default: 10).
        lr (float): The learning rate for the optimizer (default: 1e-3).

    Returns:
        None
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"MC_Dropout_{timestamp}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    model = SimpleCNNRegressionModelDropout(**model_parameters).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    loss_fn = GaussianNLLLoss()

    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in tqdm(range(num_epochs)):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            training_step(model, images, targets, loss_fn, optimizer, writer, epoch)
        
        v_images, v_targets = next(iter(val_loader))
        val_loss = validation_step(model, v_images.to(device), v_targets.to(device), loss_fn, writer, epoch)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"Best validation loss: {best_val_loss:.4f}")
        os.makedirs(config.PATH_TO_SAVE_MODEL_DIR, exist_ok=True)
        save_path = os.path.join(config.PATH_TO_SAVE_MODEL_DIR, f"{run_name}.pth")
        torch.save(model.state_dict(), save_path)
    return MC_Dropout(model, n_samples=config.NUM_SAMPLES)

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
        images, targets = images.to(device), targets.to(device)
        mean_pred, var_pred = mc_dropout.predict(images)

        loss = loss_fn(mean_pred, var_pred, targets)
        total_test_loss += loss.item()

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
        "test_loss": avg_test_loss,
        "pred_means": torch.cat(all_means),
        "pred_vars": torch.cat(all_vars),
        "targets": torch.cat(all_targets)
    }

if __name__ == "__main__":
    set_seed(config.SEED)
    os.makedirs(config.PATH_TO_RESULTS_DIR, exist_ok=True)

    train_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_TRAIN_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_VALIDATION_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_TEST_DATA_DIR, include_depth=config.INCLUDE_DEPTH, world=True, bounds=config.OUTPUT_BOUNDS), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    model = train_mc_dropout(train_loader, val_loader, model_parameters=config.MODEL_PARAMETERS, num_epochs=config.EPOCHS, lr=config.LR)
    
    test_results = test_mc_dropout(model, test_loader)

    save_path = os.path.join(config.PATH_TO_RESULTS_DIR, f"test_results_{datetime.now().strftime('%Y-%m-%d')}.json")
    with open(save_path, "w") as f:
        json.dump([], f)  # Clear the file before appending results
    
    print(f"Test Loss: {test_results['test_loss']:.4f}")
    print(f"\n--- Final Results ---")
    for i in range(test_results['pred_means'].shape[0]):
        pred = test_results['pred_means'][i].numpy()
        unc = np.sqrt(test_results['pred_vars'][i].numpy())

        target = test_results['targets'][i].numpy()
        print(f"Sample {i:<3} | Pred: [{pred[0]:>5.1f}, {pred[1]:>5.1f}] ± [{unc[0]:>5.1f}, {unc[1]:>5.1f}] mm | Target: [{target[0]:>5.1f}, {target[1]:>5.1f}]")
        
        current_file = json.load(open(save_path, "r"))
        current_file.append({
            "prediction": pred.tolist(),
            "uncertainty": unc.tolist(),
            "target": target.tolist()
        })
        with open(save_path, "w") as f:
            json.dump(current_file, f, indent=4)