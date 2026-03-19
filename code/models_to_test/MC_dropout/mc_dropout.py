import torch

class MC_Dropout(torch.nn.Module):
    def __init__(self, model, dropout_rate=0.5):
        super(MC_Dropout, self).__init__()
        self.model = model
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # Enable dropout during inference
        self.model.train()
        return self.model(x)

    def predict(self, x, n_samples=100):
        # Collect predictions from multiple forward passes
        predictions = []
        for _ in range(n_samples):
            predictions.append(self.forward(x).unsqueeze(0))
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