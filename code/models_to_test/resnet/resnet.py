import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# 1. Load the pre-trained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 2. Identify the input features of the existing FC layer
num_ftrs = model.fc.in_features

# 3. Replace the FC layer for regression (2 outputs: x and y)
# No activation function is needed at the end for raw coordinate regression
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
