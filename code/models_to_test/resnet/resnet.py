import torch
import torch.nn.functional as F
import torchvision.models as models

from models_to_test.resnet import config

class DeepEnsemble(torch.nn.Module):
    """
    A simple implementation of a deep ensemble model for regression.
    
    Args:
        models (list[torch.nn.Module]): A list of PyTorch models to be included in the ensemble.
    """
    def __init__(self, models: list[torch.nn.Module]):
        super(DeepEnsemble, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if len(models) < 2:
            raise ValueError("Ensemble requires at least two models.")

        self.models = torch.nn.ModuleList(models)
        for model in self.models:
            model.to(self.device)
            model.eval()

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the output for the given input using the ensemble of models.
         Args:
            X (torch.Tensor): Input tensor the same shape as the input expected by each model in the ensemble.
         Returns:
            tuple[torch.Tensor, torch.Tensor]: The ensemble prediction and uncertainty (mean and variance of the predictions from the individual models).
        """
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            means = []
            vars_ = []
            
            for model in self.models:
                mean, variance = model(X)
                means.append(mean)
                vars_.append(variance)
                
            means = torch.stack(means)
            vars_ = torch.stack(vars_)

            # Ensemble Mean calculation
            ensemble_mean = means.mean(dim=0)
            
            # Ensemble Variance calculation (total uncertainty = aleatoric + epistemic)
            ensemble_var = (vars_ + means**2).mean(dim=0) - ensemble_mean**2

            return ensemble_mean, ensemble_var

class ResNetRegressionModel(torch.nn.Module):
    """
    A pretrained ResNet model adapted for regression tasks with aleatoric uncertainty.
    Only using the first few layers of ResNet as a feature extractor and adding custom fully connected layers on top to output both mean and variance.
        Args:
            input_dims (tuple[int]): The dimensions of the input tensor (channels, height, width). Default is (3, 370, 250).
            output_size (int): The number of regression targets. Default is 2 (e.g., x and y coordinates).
            freeze_backbone (bool): Whether to freeze the weights of the ResNet backbone during training. Default is True.
    """
    def __init__(self, input_dims: tuple[int] = (3, 370, 250), output_size: int = 2, freeze_backbone: bool = True):
        super(ResNetRegressionModel, self).__init__()
        self.output_size = output_size
        self.input_dims = input_dims

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if self.input_dims[0] != 3:
            old_conv = self.resnet.conv1
            
            # Create a new convolutional layer with the correct number of input channels
            self.resnet.conv1 = torch.nn.Conv2d(
                in_channels=self.input_dims[0],
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            # Keep the pretrained weights for the first 3 channels and initialize the rest randomly
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3, :, :] = old_conv.weight
        
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-(config.NUMBER_OF_RES_BLOCKS-10)])

        if freeze_backbone:
            for name, param in self.feature_extractor.named_parameters():
                if "conv1" not in name and "layer4" not in name: # Don't freeze our custom input layer or the last layer of the backbone
                    param.requires_grad = False
                
        # Addapt the final fully connected layer (and averagepooling before it) to output both mean and variance
        dummy_input = torch.zeros(1, self.input_dims[0], self.input_dims[1], self.input_dims[2])
        dummy_features = self.feature_extractor(dummy_input)
        flattened_size = dummy_features.view(1, -1).size(1)
        
        self.fc = torch.nn.Linear(flattened_size, output_size * 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # Split the output into mean and variance components
        mean = x[:, :self.output_size]
        raw_variance = x[:, self.output_size:]

        # Ensure variance is positive and add small value for numerical stability
        variance = F.softplus(raw_variance) + 1e-2
        
        return mean, variance