import torch

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
            predictions = []
            for model in self.models:
                pred = model(X)
                predictions.append(pred)
            stacked = torch.stack(predictions)
            return stacked.mean(dim=0), stacked.var(dim=0)


class SimpleCNNRegressionModel(torch.nn.Module):
    """
    A simple convolutional network for regression tasks.
    
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

        # Global Average Pooling to transition from CNN to fully connected layer
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer for regression output
        self.fc = torch.nn.Linear(hidden_channels[-1], output_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x