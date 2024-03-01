from datetime import timedelta
from typing import Any, Literal, Sequence
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import numpy as np
import config
from utils.ops import load_sb_image, save_geotiff
from einops import rearrange


class SavePrediction(BasePredictionWriter):
    def __init__(self, tiff_path, n_prev, patch_size, test_times, border_removal,**kwargs):
        super().__init__(write_interval = 'batch_and_epoch')
        self.tiff_path = tiff_path
        self.patch_size = patch_size
        self.n_prev = n_prev
        self.border_removal = border_removal
        
        mask = load_sb_image(config.path_to_mask)
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
        save_geotiff(config.path_to_mask, self.tiff_path, self.final_image, 'float')