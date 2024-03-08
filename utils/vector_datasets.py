from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import fiona
import numpy as np
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange
from skimage.util import view_as_windows
from lightning import LightningDataModule
import paths
import albumentations as A
from albumentations.augmentations.geometric.transforms import HorizontalFlip, VerticalFlip 

class VectorDataModule(LightningDataModule):
    def __init__(self,
                 n_previous_times,
                 train_times,
                 val_times,
                 test_times,
                 train_batch_size,
                 train_num_workers,
                 features
                 ) -> None:
        super().__init__()
        self.n_previous_times = n_previous_times
        self.train_times = train_times
        self.val_times = val_times
        self.test_times = test_times
        self.features = features
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        
        mask = load_sb_image(paths.path_to_mask).flatten()
        
        #deforestation data
        qd_paths_dict = paths.path_to_data['def_data']
        def_data = load_ml_image(qd_paths_dict)
        def_data = rearrange(def_data, 'h w c -> (h w) c')[mask == 1]
        self.data_max = def_data.max()
        
        train_data = def_data[:, n_previous_times:train_times]
        val_data = def_data[:, n_previous_times + train_times:train_times+val_times]
        
        train_prop = (train_data == 0).sum() / (train_data > 0).sum()
        val_prop = (val_data == 0).sum() / (val_data > 0).sum()
        
        self.train_weights = np.ones_like(train_data.flatten())
        self.val_weights = np.ones_like(val_data.flatten())
        
        train_data = rearrange(train_data, 'n c -> (c n)')
        val_data = rearrange(val_data, 'n c -> (c n)')
        
        self.train_weights[train_data>0] = train_prop
        self.val_weights[val_data>0] = val_prop
        
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def train_dataloader(self):
        train_ds = VectorTrainDataset(
            n_prev = self.n_previous_times,
            band_first = 0,
            band_last = self.train_times,
            features = self.features,
            mode='train'
        )
        return DataLoader(
            train_ds,
            batch_size = self.train_batch_size,
            #shuffle = True,
            sampler=WeightedRandomSampler(self.train_weights, len(train_ds)),
            num_workers = self.train_num_workers
        )
    
    def val_dataloader(self):
        val_ds = VectorTrainDataset(
            n_prev = self.n_previous_times,
            band_first = self.train_times,
            band_last = self.train_times + self.val_times,
            features = self.features,
            mode='validation'
        )
        return DataLoader(
            val_ds,
            batch_size = self.train_batch_size,
            sampler=WeightedRandomSampler(self.val_weights, len(val_ds)),
            num_workers = self.train_num_workers
        )
    
    def test_dataloader(self):
        test_ds = VectorTrainDataset(
            n_prev = self.n_previous_times,
            band_first = self.train_times + self.val_times,
            band_last = self.train_times + self.val_times + self.test_times,
            features = self.features,
            mode='test'
        )
        return DataLoader(
            test_ds,
            batch_size = self.train_batch_size,
            num_workers = self.train_num_workers
        )
    
    def predict_dataloader(self):
        prediction_ds = VectorPredictionDataset(
            n_prev = self.n_previous_times,
            band_first = self.train_times + self.val_times,
            band_last = self.train_times + self.val_times + self.test_times,
            features = self.features,
        )
        return DataLoader(
            prediction_ds,
            batch_size = self.train_batch_size,
            num_workers = self.train_num_workers
        )
    


class VectorTrainDataset(Dataset):
    def __init__(self, n_prev, band_first, band_last, features, mode, ):
        self.n_prev = n_prev
        self.band_first = band_first
        self.band_last = band_last
        self.mode = mode
        
        mask = load_sb_image(paths.path_to_mask).flatten()
        
        #deforestation data
        qd_paths_dict = paths.path_to_data['def_data']
        def_data = load_ml_image(qd_paths_dict)[:,:,band_first:band_last]
        self.def_data = rearrange(def_data, 'h w c -> (h w) c')[mask == 1]
        
        

    def __len__(self):
        return self.def_data.shape[0]*(self.band_last - self.band_first - self.n_prev)
    
    def __getitem__(self, index):
        vector_i = index % self.def_data.shape[0]
        band_i = index // self.def_data.shape[0] + self.n_prev
        
        label = np.expand_dims(self.def_data, axis=-1)[vector_i, band_i]
        
        data = self.def_data[vector_i, band_i-self.n_prev:band_i]
        
        weight = 1
        if label>0:
            weight = 10 * weight
        
        return data/100, label, weight


class VectorPredictionDataset(Dataset):
    def __init__(self, n_prev, band_first, band_last, features):
        self.n_prev = n_prev
        self.band_first = band_first
        self.band_last = band_last
        
        self.mask = load_sb_image(paths.path_to_mask).flatten()
        
        #deforestation data
        qd_paths_dict = paths.path_to_data['def_data']
        def_data = load_ml_image(qd_paths_dict)[:,:,band_first:band_last]
        self.def_data = rearrange(def_data, 'h w c -> (h w) c')

    def __len__(self):
        return self.def_data.shape[0]*(self.band_last - self.band_first - self.n_prev)
    
    def __getitem__(self, index):
        vector_i = index % self.def_data.shape[0]
        band_i = index // self.def_data.shape[0] + self.n_prev
        
        label = self.def_data[vector_i, band_i]
        
        data = self.def_data[vector_i, band_i-self.n_prev:band_i]
        
        weight = self.mask[vector_i]

        
        return data/100, label, weight, vector_i, band_i