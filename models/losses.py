from torch import nn
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
import torch

class RegressionFocalLoss(_Loss):
    def __init__(self, alpha, beta, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        
    def forward(self, y_pred, y_true):
        mae = F.l1_loss(y_pred, y_true, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction) / self.beta
        loss = torch.pow(mae, self.alpha)*F.mse_loss(y_pred, y_true, size_average=self.size_average, reduce=self.reduce, reduction=self.reduction)
        return loss
        


