import torch
import torch.nn as nn
import torch.nn.functional as F 


class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.tensor, target: torch.tensor):
        loss = (input - target) ** 2
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss 

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight = None, reduction: str = "mean"):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.tensor, target: torch.tensor):
        log_probs = F.log_softmax(input, dim=-1)
        if self.weight is not None:
            log_probs = log_probs * self.weight
        loss = - torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss 

class BCELoss(nn.Module):
    def __init__(self, weight = None, reduction: str = "mean"):
        super(BCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.tensor, target: torch.tensor):
        loss = - (target * torch.log(input) + (1 - target) * torch.log(1 - input))
        if self.weight is not None:
            loss = loss * self.weight 
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss 
        