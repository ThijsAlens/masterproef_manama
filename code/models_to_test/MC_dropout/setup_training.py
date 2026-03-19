import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import random
import numpy as np

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

def get_data_loader(data_path: str, input_dimension: tuple=(3, 64, 64), augment: bool=False, batch_size: int=32):
    """
    Creates a data loader for the specified dataset.

    Args:
        data_path (str): Path to the dataset.
        input_dimension (tuple): Desired input size for the model (default: (3, 64, 64)).
        augment (bool): Whether to apply data augmentation (default: False).
        batch_size (int): Number of samples per batch (default: 32).

    Returns:
        DataLoader: A data loader for the specified dataset.
    """
    # Standard normalization values for ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if augment:
        data_transform = transforms.Compose([
            transforms.Resize(input_dimension),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(input_dimension),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    # Load datasets
    dataset = datasets.ImageFolder(root=data_path, transform=data_transform)

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )

    print(f"Classes found: {dataset.classes}")
    return loader

def training_step(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, writer: torch.utils.tensorboard.SummaryWriter, epoch: int) -> float:
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
    
    outputs = model(images)
    loss = loss_fn(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar(f"{loss_fn.__class__.__name__}/Train", loss.item(), epoch)
    return loss.item()

def validation_step(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module, writer: torch.utils.tensorboard.SummaryWriter, epoch: int) -> float:
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
        outputs = model(images)
        val_loss = loss_fn(outputs, targets)
        writer.add_scalar(f"{loss_fn.__class__.__name__}/Val", val_loss.item(), epoch)
    return val_loss.item()

def train_mc_dropout(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, writer: torch.utils.tensorboard.SummaryWriter, num_epochs: int=10) -> None:
    """
    Train a model using Monte Carlo Dropout.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        loss_fn (torch.nn.Module): The loss function to compute training and validation losses.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer to log training and validation metrics.
        num_epochs (int): The number of epochs to train the model (default: 10).

    Returns:
        None
    """
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, targets in train_loader:
            train_loss += training_step(model, images, targets, loss_fn, optimizer, writer, epoch)
        
        val_loss = 0.0
        for images, targets in val_loader:
            val_loss += validation_step(model, images, targets, loss_fn, writer, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")
    return

def test_mc_dropout(model: torch.nn.Module, test_loader: DataLoader, writer: torch.utils.tensorboard.SummaryWriter, epoch: int) -> None:
    """
    Test a model using Monte Carlo Dropout.

    Args:
        model (torch.nn.Module): The model to test.
        test_loader (DataLoader): The data loader for the test dataset.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer to log test metrics.
        epoch (int): The current epoch number for logging purposes.

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            predictions = model.predict(images)
            pred_mean = predictions.mean(dim=0)
            pred_var = predictions.var(dim=0)
            writer.add_scalar("Predictions/Mean", pred_mean.item(), epoch)
            writer.add_scalar("Predictions/Variance", pred_var.item(), epoch)
    return