import torch
import torch.nn.functional as F

class GaussianNLLLoss(torch.nn.Module):
    """
    Negative Log-Likelihood loss for a Gaussian distribution, 
    as described in Lakshminarayanan et al. (2017).
    """
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()

    def forward(self, mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * torch.log(variance) + 0.5 * ((target - mean) ** 2) / variance
        return loss.mean()