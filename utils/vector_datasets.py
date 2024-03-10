from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange
from lightning import LightningDataModule
from features import features, mask_path, FeatureData, FeatureDataSet, get_first_lag

class VectorDataModule(LightningDataModule):
    def __init__(self,
                 n_previous_times,
                 train_times,
                 val_times,
                 test_times,
                 train_batch_size,
                 train_num_workers,
                 features_list,
                 normalize_data,
                 normalize_label
                 ) -> None:
        super().__init__()
        self.n_previous_times = n_previous_times
        self.train_times = train_times
        self.val_times = val_times
        self.test_times = test_times
        self.features_list = features_list
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.normalize_data = normalize_data
        self.normalize_label = normalize_label
        
        #mask = load_sb_image(mask_path).flatten()
        
        #deforestation data
        self.first_lag = get_first_lag(features_list)
               
        #train_data = def_data.data[n_previous_times + self.first_lag : train_times]
        train_data = FeatureData(features_list[0], masked=True)
        train_data.filter_period(n_previous_times + self.first_lag, train_times)
        #val_data = def_data.data[n_previous_times + train_times : train_times + val_times]
        val_data = FeatureData(features_list[0], masked=True)
        val_data.filter_period(n_previous_times + train_times, train_times + val_times)
        
        train_prop = (train_data.data == 0).sum() / (train_data.data > 0).sum()
        val_prop = (val_data.data == 0).sum() / (val_data.data > 0).sum()
        
        self.train_weights = np.ones_like(train_data.data.flatten())
        self.val_weights = np.ones_like(val_data.data.flatten())
        
        train_data = rearrange(train_data.data, 'l n -> (l n)')
        val_data = rearrange(val_data.data, 'l n -> (l n)')
        
        self.train_weights[train_data>0] = train_prop
        self.val_weights[val_data>0] = val_prop
        
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def train_dataloader(self):
        train_ds = VectorTrainDataset(
            n_prev = self.n_previous_times,
            first_lag = self.first_lag,
            last_lag = self.train_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label
        )
        return DataLoader(
            train_ds,
            batch_size = self.train_batch_size,
            #shuffle = True,
            sampler=WeightedRandomSampler(self.train_weights, len(train_ds)),
            num_workers = self.train_num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        val_ds = VectorTrainDataset(
            n_prev = self.n_previous_times,
            first_lag = self.train_times,
            last_lag = self.train_times + self.val_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label
        )
        return DataLoader(
            val_ds,
            batch_size = self.train_batch_size,
            sampler=WeightedRandomSampler(self.val_weights, len(val_ds)),
            persistent_workers=True,
            num_workers = self.train_num_workers
        )
    
    def test_dataloader(self):
        test_ds = VectorTrainDataset(
            n_prev = self.n_previous_times,
            first_lag = self.train_times + self.val_times,
            last_lag = self.train_times + self.val_times + self.test_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label
        )
        return DataLoader(
            test_ds,
            batch_size = self.train_batch_size,
            #num_workers = self.train_num_workers
        )
    
    def predict_dataloader(self):
        self.prediction_ds = VectorPredictionDataset(
            n_prev = self.n_previous_times,
            first_lag = self.train_times + self.val_times,
            last_lag = self.train_times + self.val_times + self.test_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label
        )
        return DataLoader(
            self.prediction_ds,
            batch_size = self.train_batch_size,
            #num_workers = self.train_num_workers
        )
    


class VectorTrainDataset(Dataset):
    def __init__(self, n_prev, first_lag, last_lag, features_list, normalize_data, normalize_label):
        self.n_prev = n_prev
        self.first_lag = first_lag
        self.last_lag = last_lag
        self.features_list = features_list
        
        self.dataset = FeatureDataSet(features_list, masked=True, normalize_data=normalize_data, normalize_label=normalize_label)
        self.dataset.filter_period(first_lag, last_lag)

    def __len__(self):
        return (self.dataset.n_lags() - self.n_prev) * (self.dataset.n_vectors())
    
    def __getitem__(self, index):
        lag_i = (index // self.dataset.n_vectors()) + self.n_prev
        vector_i = index % self.dataset.n_vectors()
        
        data, label = self.dataset.get_data(lag_i, vector_i, self.n_prev)
        
        if label > 0 and label < 30 :
            weight = 100
        elif label >= 30:
            weight = 10000
        else:
            weight = 1
            
        return data, label, weight


class VectorPredictionDataset(Dataset):
    def __init__(self, n_prev, first_lag, last_lag, features_list, normalize_data, normalize_label):
        self.n_prev = n_prev
        self.first_lag = first_lag
        self.last_lag = last_lag
        self.features_list = features_list
        
        self.mask = load_sb_image(mask_path).flatten()
        
        self.dataset = FeatureDataSet(features_list, masked=False, normalize_data=normalize_data, normalize_label=normalize_label)
        self.dataset.filter_period(first_lag, last_lag)

    def __len__(self):
        return (self.dataset.n_lags() - self.n_prev) * (self.dataset.n_vectors())
    
    def __getitem__(self, index):
        lag_i = (index // self.dataset.n_vectors()) + self.n_prev
        vector_i = index % self.dataset.n_vectors()
        mask_i = self.mask[vector_i]
        
        data, label = self.dataset.get_data(lag_i, vector_i, self.n_prev)
        
        return data, label, mask_i, vector_i, lag_i