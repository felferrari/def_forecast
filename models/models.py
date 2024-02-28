from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import segmentation_models_pytorch as smp
import lightning as L
import torch
import torch.nn.functional as F
from pydoc import locate

class Model1(L.LightningModule):
    def __init__(self, criterion, optimizer, in_channels, lr, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = smp.UnetPlusPlus(
            in_channels=in_channels,
            classes = 1,
            encoder_weights = None
        )
        self.criterion = criterion
        self.lr = lr
        self.optimizer = optimizer
   
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, mask = batch
        y_hat = self.model(x)
        #loss = F.mse_loss(y_hat, y)
        loss = self.criterion(y_hat, y)
        loss = (loss*mask).sum() / mask.sum()
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, mask = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        loss = (loss*mask).sum() / mask.sum()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        Optmizer = locate(self.optimizer['optimizer'])
        return Optmizer(self.parameters(), **self.optimizer['parameters'])

