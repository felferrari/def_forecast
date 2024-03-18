from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import torch
import math
import mlflow

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
        
        self.train_max, self.train_min = -math.inf, math.inf 
        self.val_max, self.val_min = -math.inf, math.inf
        self.test_max, self.test_min = -math.inf, math.inf
        self.pred_max, self.pred_min = -math.inf, math.inf

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, weight, lag_i, vec_i = batch
        y_hat = self.model(x) 
        #loss = F.mse_loss(y_hat, y)
        loss = self.criterion(y_hat, y)
        loss = (loss*weight).sum() / weight.sum()
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.train_min = min(self.train_min, lag_i.min().item())
        self.train_max = max(self.train_max, lag_i.max().item())
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log('train_lag_max', self.train_max)
        self.log('train_lag_min', self.train_min)
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, weight, lag_i, vec_i = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        loss = (loss*weight).sum() / weight.sum()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.val_min = min(self.val_min, lag_i.min().item())
        self.val_max = max(self.val_max, lag_i.max().item())
        return loss
    
    def on_validation_epoch_end(self) -> None:
        self.log('val_lag_max', self.val_max)
        self.log('val_lag_min', self.val_min)
        
    def on_test_start(self) -> None:
        self.test_max, self.test_min = -math.inf, math.inf
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, weight, lag_i, vec_i = batch
        y_hat = self.model(x)
        #y_hat = torch.zeros_like(y_hat)
        loss = self.criterion(y_hat, y)
        loss = (loss*weight).sum() / weight.sum()
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.test_min = min(self.test_min, lag_i.min().item())
        self.test_max = max(self.test_max, lag_i.max().item())
        
        y_f = y.flatten()
        y_hat_f = y_hat.flatten()
        mask_f = weight.flatten()
        
        y_f = y_f[mask_f > 0]
        y_hat_f = y_hat_f[mask_f > 0]
        
        self.test_mse.update(y_hat_f, y_f)
        self.test_mae.update(y_hat_f, y_f)
        return loss
    
    def on_test_epoch_end(self) -> None:
        self.log('test_lag_max', self.test_max)
        self.log('test_lag_min', self.test_min)
        self.log('test_mse', self.test_mse.compute())
        self.log('test_mae', self.test_mae.compute())
        self.test_mse.reset()
        self.test_mae.reset()
        
    def on_predict_start(self) -> None:
        self.pred_max, self.pred_min = -math.inf, math.inf
    
    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, weight, lag_i, vec_i = batch
        y_pred = self.model(x) 
        self.pred_min = min(self.pred_min, lag_i.min().item())
        self.pred_max = max(self.pred_max, lag_i.max().item())
        return y_pred 
    
    def on_predict_epoch_end(self) -> None:
        mlflow.log_metric('pred_lag_max', self.pred_max)
        mlflow.log_metric('pred_lag_min', self.pred_min)
        

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), **self.optimizer_params)

