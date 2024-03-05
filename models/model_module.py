from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import torch

class ModelModule(L.LightningModule):
    def __init__(self, model, criterion, optimizer, optimizer_params, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        self.optimizer_params = optimizer_params
        self.optimizer = optimizer
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        
        
   
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, mask = batch
        y_hat = self.model(x) 
        #loss = F.mse_loss(y_hat, y)
        loss = self.criterion(y_hat, y)
        loss = (loss*mask).sum() / mask.sum()
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, mask = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        loss = (loss*mask).sum() / mask.sum()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, mask = batch
        y_hat = self.model(x)
        #y_hat = torch.zeros_like(y_hat)
        loss = self.criterion(y_hat, y)
        loss = (loss*mask).sum() / mask.sum()
        #self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        
        y_f = y.flatten()
        y_hat_f = y_hat.flatten()
        mask_f = mask.flatten()
        
        y_f = y_f[mask_f > 0]
        y_hat_f = y_hat_f[mask_f > 0]
        
        self.test_mse.update(y_hat_f, y_f)
        self.test_mae.update(y_hat_f, y_f)
        return loss
    
    def on_test_epoch_end(self) -> None:
        self.log('test_mse', self.test_mse.compute())
        self.log('test_mae', self.test_mae.compute())
        self.test_mse.reset()
        self.test_mae.reset()
        
    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, mask, idx, band_i = batch
        y_pred = self.model(x) 
        y_pred [ mask == 0] = 0
        return y_pred 
        

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_params)

