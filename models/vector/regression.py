from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import torch
import math
import mlflow
from torch import nn
from ..layers import TransformerDecoder


class RegressionModelModule(L.LightningModule):
    def __init__(self, model, criterion, optimizer, optimizer_params, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        self.optimizer_params = optimizer_params
        self.optimizer = optimizer
        
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

class MlpRegression(nn.Module):
    def __init__(self, layers, input_sample, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        input_size = 0
        for k in input_sample[0]:
            input_size += len(input_sample[0][k])
        self.mlp = nn.Sequential()
        self.mlp.append(nn.Linear(in_features=input_size, out_features=layers[0]))
        self.mlp.append(nn.BatchNorm1d(num_features=layers[0]))
        self.mlp.append(nn.ReLU())
        for i in range(len(layers)-1):
            self.mlp.append(nn.Linear(in_features=layers[i], out_features=layers[i+1]))
            self.mlp.append(nn.BatchNorm1d(num_features=layers[i+1]))
            self.mlp.append(nn.ReLU())
            
        self.mlp.append(nn.Linear(in_features=layers[-1], out_features=1))
        self.mlp.append(nn.ReLU())
        
        
    def forward(self, x_dict):
        x = torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        x = self.mlp(x)
        return x
    

class TransformerRegression(nn.Module):
    def __init__(self, input_sample, n_layers, d_model, n_head, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_dim = 0
        for sample_k in input_sample[0]:
            in_dim += input_sample[0][sample_k].shape[0]
        self.proj = nn.Linear(in_features=in_dim, out_features=d_model)
        self.trans_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head
            )
            for _ in range(n_layers)
        ])
        self.decoder = TransformerDecoder(in_dim=d_model, out_dim=1, activation=nn.ReLU())
        
    def forward(self, x_dict):
        x = torch.cat([x_dict[x_k] for x_k in x_dict], dim = -1)
        x = self.proj(x)
        for layer in self.trans_layers:
            x = layer(x)
        x = self.decoder(x)
        return x