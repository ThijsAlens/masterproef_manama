import torch

from models_to_test.MC_dropout import config

class MC_Dropout(torch.nn.Module):
    """
    A simple implementation of Monte Carlo Dropout for regression tasks.
    
    Args:
        model (torch.nn.Module): A PyTorch model to be used for regression. The model is assumed to have dropout layers.
        n_samples (int): The number of Monte Carlo samples to use for prediction. Default is 100.
    """
    def __init__(self, model: torch.nn.Module, n_samples: int = 100):
        super(MC_Dropout, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = model.to(self.device)
        self.n_samples = n_samples
    def predict(self, x):
        """
        Predict the output for the given input using Monte Carlo Dropout.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The ensemble prediction and uncertainty (mean and variance of the predictions from the individual models).
        """
        x = x.to(self.device)
        self.model.train()  # Set the model to training mode to enable dropout during inference
        with torch.no_grad():
            means = []
            vars_ = []
            for _ in range(self.n_samples):
                mean, variance = self.model(x)
                means.append(mean)
                vars_.append(variance)

            means = torch.stack(means)
            vars_ = torch.stack(vars_)

            # Ensemble Mean calculation
            ensemble_mean = means.mean(dim=0)
            
            # Ensemble Variance calculation (total uncertainty = aleatoric + epistemic)
            ensemble_var = (vars_ + means**2).mean(dim=0) - ensemble_mean**2

            return ensemble_mean, ensemble_var

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