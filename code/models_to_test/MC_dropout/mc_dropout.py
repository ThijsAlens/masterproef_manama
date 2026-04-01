import torch

class MC_Dropout(torch.nn.Module):
    """
    A simple implementation of Monte Carlo Dropout for regression tasks.
    
    Args:
        model (torch.nn.Module): A PyTorch model to be used for regression. The model is assumed to have dropout layers.
    """
    def __init__(self, model: torch.nn.Module):
        super(MC_Dropout, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = model.to(self.device)

    def predict(self, x, n_samples=100):
        """
        Predict the output for the given input using Monte Carlo Dropout.
        
        Args:
            x (torch.Tensor): Input tensor.
            n_samples (int): The number of Monte Carlo samples to use for prediction.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The ensemble prediction and uncertainty (mean and variance of the predictions from the individual models).
        """
        x = x.to(self.device)
        self.model.train()  # Set the model to training mode to enable dropout during inference
        with torch.no_grad():
            predictions = []
            for _ in range(n_samples):
                predictions.append(self.model(x))
            stacked = torch.stack(predictions)
            return stacked.mean(dim=0), stacked.var(dim=0)
    
class SimpleCNNRegressionModelDropout(torch.nn.Module):
    """
    A simple convolutional network for regression tasks. Dropout layers are included between all convolutional layers.
    
    Args:
        input_size (int): The number of input features.
        hidden_size (list[int]): The number of hidden units in the hidden layer. Default is [16].
        output_size (int): The number of output features.
        p_list (list[float]): The dropout probabilities to be applied after each convolutional layer. Default is [0.5].
    """
    
    def __init__(self, input_channels: int = 3, hidden_channels: list[int] = [16], output_size: int = 2, p_list: list[float] = [0.5]):
        super(SimpleCNNRegressionModelDropout, self).__init__()

        if len(p_list) != len(hidden_channels):
            raise ValueError("The length of p_list must match the length of hidden_channels.")

        # Convolutional Layers (with ReLU and MaxPool and Dropout)
        self.conv_layers = []
        current_channels = input_channels
        for i, h_channels in enumerate(hidden_channels):
            self.conv_layers.append(torch.nn.Conv2d(current_channels, h_channels, kernel_size=3, padding=1))
            self.conv_layers.append(torch.nn.ReLU())
            self.conv_layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv_layers.append(torch.nn.Dropout2d(p=p_list[i]))
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