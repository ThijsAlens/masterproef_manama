import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

from ensemble import DeepEnsemble, SimpleCNNRegressionModel
import config
from dataset.custom_dataset import CustomDataset 

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

def train_ensemble_bagging(num_models: int=5, training_dataset: DataLoader=None, validation_dataset: DataLoader=None, model_parameters: dict[str, any]=None, epochs: int=10, lr:int=1e-3, batch_size: int=32, sample_ratio: float=1.0) -> DeepEnsemble:
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
        sample_ratio (float): The ratio of the dataset to sample for each model (between 0 and 1).
    
    Returns:
        DeepEnsemble: A trained deep ensemble model.
    """
    if model_parameters is None:
        raise ValueError("model_parameters must be provided to initialize the models in the ensemble.")
    
    if training_dataset is None or validation_dataset is None:
        raise ValueError("training_dataset and validation_dataset must be provided to train the models in the ensemble.")
    
    ensemble_models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    for i in range(num_models):
        # Create a unique TensorBoard run name for this ensemble member
        run_name = f"ensemble_model_{i}_{timestamp}"
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        
        model = SimpleCNNRegressionModel(**model_parameters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        
        # Bagging indices
        indices = torch.randint(0, len(training_dataset), (len(training_dataset),)).tolist()
        train_loader = DataLoader(Subset(training_dataset, indices), batch_size=batch_size, shuffle=True)

        print(f"Training Model {i}...")
        for epoch in tqdm(range(epochs)):
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                training_step(model, images, targets, loss_fn, optimizer, writer, epoch)
            
            v_images, v_targets = next(iter(validation_dataset))
            validation_step(model, v_images.to(device), v_targets.to(device), loss_fn, writer, epoch)
            
        ensemble_models.append(model)
        writer.close()

    return DeepEnsemble(ensemble_models)

def test_ensemble(ensemble: DeepEnsemble, test_dataset: DataLoader, loss_fn=torch.nn.MSELoss()) -> dict[str, any]:
    """
    Test the performance of the trained ensemble on a test dataset.

    Args:
        ensemble (DeepEnsemble): The trained ensemble model to test.
        test_dataset (DataLoader): The dataset to evaluate the ensemble on.
        loss_fn (torch.nn.Module): The loss function to compute the test loss (default: MSELoss).

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
        loss = loss_fn(mean_pred, targets)
        
        total_test_loss += loss.item()
        
        all_means.append(mean_pred.cpu())
        all_vars.append(var_pred.cpu())
        all_targets.append(targets.cpu())

    avg_test_loss = total_test_loss / len(test_dataset)
    
    return {
        "loss": avg_test_loss,
        "predictions": torch.cat(all_means),
        "uncertainty": torch.cat(all_vars),
        "targets": torch.cat(all_targets)
    }

if __name__ == "__main__":
    set_seed(config.SEED)

    train_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_TRAIN_DATA_DIR, include_depth=True, world=True), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_VALIDATION_DATA_DIR, include_depth=True, world=True), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(CustomDataset(source_dir=config.PATH_TO_TEST_DATA_DIR, include_depth=True, world=True), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    deep_ensemble = train_ensemble_bagging(num_models=config.NUM_MODELS, training_dataset=train_loader, validation_dataset=val_loader, model_parameters=config.MODEL_PARAMETERS, epochs=config.EPOCHS, lr=config.LR, batch_size=config.BATCH_SIZE, sample_ratio=config.BAGGING_SAMPLE_RATIO)

    results = test_ensemble(deep_ensemble, test_loader)
    print(f"Test Loss: {results['loss']:.4f}")
    for i in range(results['predictions'].shape[0]):
        print(f"Sample {str(i):<5} | Prediction = {str(results['predictions'][i].numpy()):<10} | Uncertainty = {str(results['uncertainty'][i].numpy()):<10} | Target = {str(results['targets'][i].numpy()):<10}")