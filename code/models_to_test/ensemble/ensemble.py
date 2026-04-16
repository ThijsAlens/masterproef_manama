import torch

from models_to_test.ensemble import config

class DeepEnsemble:
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

        self.models = models
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


class SimpleCNNRegressionModel(torch.nn.Module):
    """
    A simple convolutional network for regression tasks with the capability to model aleatoric uncertainty.
    It outputs both the mean and variance for each prediction.
    
    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units in the hidden layer.
        output_size (int): The number of output features.
    """
    
    def __init__(self, input_channels: int = 3, hidden_channels: list[int] = [16], output_size: int = 2):
        super(SimpleCNNRegressionModel, self).__init__()
        
        # Convolutional Layers (with ReLU and MaxPool)
        self.conv_layers = []
        current_channels = input_channels
        for h_channels in hidden_channels:
            self.conv_layers.append(torch.nn.Conv2d(current_channels, h_channels, kernel_size=3, padding=1))
            self.conv_layers.append(torch.nn.ReLU())
            self.conv_layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = h_channels
        self.conv_layers = torch.nn.Sequential(*self.conv_layers)

        # Flatten the output of the convolutional layers to feed into the fully connected layer
        dummy_input = torch.zeros(1, input_channels, config.INPUT_DIMENSION[1], config.INPUT_DIMENSION[2])
        dummy_output = self.conv_layers(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        
        # Fully Connected Layer for regression output
        self.fc = torch.nn.Linear(flattened_size, output_size * 2)  # Output mean and variance for each output feature (aleatoric uncertainty)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # Split the output into mean and variance components
        mean = x[:, :self.fc.out_features // 2]
        variance = x[:, self.fc.out_features // 2:]

        # Ensure variance is positive and add small value for numerical stability
        variance = torch.nn.functional.softplus(variance) + 1e-6
        return mean, variance