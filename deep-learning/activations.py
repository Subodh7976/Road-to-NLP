import torch
import torch.nn as nn
import torch.nn.functional as F


class Sigmoid(nn.Module):
    def forward(self, x: torch.tensor):
        return 1 / (1 + torch.exp(-x))
    
class Tanh(nn.Module):
    def forward(self, x: torch.tensor):
        return torch.tanh(x)
    
class ReLU(nn.Module):
    def forward(self, x: torch.tensor):
        return torch.clamp(x, min=0)
    
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.tensor):
        return torch.where(x > 0, x, self.negative_slope * x)
    
class PReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.full((num_parameters,), init))
    
    def forward(self, x: torch.tensor):
        return torch.where(x > 0, x, self.alpha * x)
    
class ELU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ELU, self).__init__()
        self.alpha = alpha 
    
    def forward(self, x: torch.tensor):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
    
class Swish(nn.Module):
    def forward(self, x: torch.tensor):
        return x * torch.sigmoid(x)
    
class Softmax(nn.Module):
    def __init__(self, dim: int = None):
        super(Softmax, self).__init__()
        self.dim = dim 
    
    def forward(self, x: torch.tensor):
        return torch.softmax(x, dim=self.dim)

