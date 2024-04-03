from datetime import timedelta
from typing import Any, Literal, Sequence
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import numpy as np
import datasets.features as features
from utils.ops import load_sb_image, save_geotiff
from einops import rearrange
import mlflow
import tempfile
import uuid

class SaveImagePrediction(BasePredictionWriter):
    def __init__(self, n_prev, patch_size, test_times, border_removal, log_tiff = False, **kwargs):
        super().__init__(write_interval = 'batch_and_epoch')
        self.patch_size = patch_size
        self.n_prev = n_prev
        self.border_removal = border_removal
        self.save_tiff = log_tiff
        
        mask = load_sb_image(features.mask)
        shape = mask.shape
        self.padded_shape = (shape[0] + 2*patch_size, shape[1] + 2*patch_size, test_times - n_prev)
        
        self.predicted_values = np.zeros(self.padded_shape, dtype=np.float64)
        self.predicted_values = rearrange(self.predicted_values, 'h w c -> (h w) c')
        self.predicted_count = np.zeros(self.padded_shape, dtype=np.uint32)
        self.predicted_count = rearrange(self.predicted_count, 'h w c -> (h w) c')
               
    
    def write_on_batch_end(self, trainer: Trainer, pl_module: LightningModule, prediction: Any, batch_indices, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        x, y, mask, idx, band_i = batch
        for sample in range(band_i.shape[0]):
            sample_pred = prediction[sample][0][self.border_removal:-self.border_removal,self.border_removal:-self.border_removal].cpu().numpy()
            sample_band_i = band_i[sample].cpu().numpy() - self.n_prev
            sample_idx = idx[sample][self.border_removal:-self.border_removal,self.border_removal:-self.border_removal].cpu().numpy()
            
            self.predicted_values[sample_idx, sample_band_i] += sample_pred
            self.predicted_count[sample_idx, sample_band_i] += np.ones_like(sample_pred, dtype = np.uint32)
        #return super().write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
    
    def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]) -> None:
        final_image = self.predicted_values / self.predicted_count
        final_image = np.nan_to_num(final_image, nan=0)
        final_image = rearrange (final_image, '(h w) c -> h w c', h = self.padded_shape[0], w = self.padded_shape[1])
        self.final_image = final_image[self.patch_size : -self.patch_size, self.patch_size : -self.patch_size]
        if self.save_tiff:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = Path(tmp_dir) / f'{mlflow.active_run().info.run_name}_{uuid.uuid4()}.tif'
                save_geotiff(features.mask, tmp_file, self.final_image, 'float')
                mlflow.log_artifact(tmp_file, 'prediction')
                
class SaveVectorPrediction(BasePredictionWriter):
    def __init__(self, test_time_0, test_times, **kwargs):
        super().__init__(write_interval = 'batch_and_epoch')
        #self.n_prev = n_prev
        self.test_time_0 = test_time_0
        
        self.mask = load_sb_image(features.mask_path)
        self.shape = self.mask.shape
        
        self.predicted_values = np.zeros(self.shape + (test_times,), dtype=np.float64)
        self.predicted_values = rearrange(self.predicted_values, 'h w c -> (h w) c')
               
    
    def write_on_batch_end(self, trainer: Trainer, pl_module: LightningModule, prediction: Any, batch_indices, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        x_list, y_list, weight_list, lag_i_list, vector_i_list = batch
        for i in range(len(vector_i_list)):
            pred_i = prediction[i]
            lag_i = lag_i_list[i]
            vector_i = vector_i_list[i]
            weight_i = weight_list[i]
            
            self.predicted_values[vector_i, lag_i- self.test_time_0] = pred_i * weight_i
    
    def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]) -> None:
        self.final_image = rearrange(self.predicted_values, '(h w) c -> h w c', h = self.shape[0], w = self.shape[1])
        #self.final_image = np.zeros_like(self.final_image)
        if self.save_tiff:
            with tempfile.TemporaryDirectory() as tmp_dir:
                cells_ones = np.ones_like(self.final_image[0,0])
                self.final_image[self.mask == 0] = -1 * cells_ones
                #tmp_file = Path(tmp_dir) / f'{mlflow.active_run().info.run_name}_{uuid.uuid4()}.tif'
                tmp_file = Path(tmp_dir) / f'{mlflow.active_run().info.run_name}.tif'
                save_geotiff(features.mask_path, tmp_file, self.final_image, 'float')
                mlflow.log_artifact(tmp_file, 'prediction')
