from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange
from lightning import LightningDataModule
from datasets.features import features, mask_path, FeatureData, FeatureDataSet #, get_first_lag

class DataModule(LightningDataModule):
    def __init__(self,
                 #n_previous_times,
                 time_0,
                 train_times,
                 val_times,
                 test_times,
                 train_batch_size,
                 train_num_workers,
                 features_list,
                 pred_batch_size,
                 normalize_data,
                 normalize_label
                 ) -> None:
        super().__init__()
        #self.n_previous_times = n_previous_times
        self.time_0 = time_0
        self.train_times = train_times
        self.val_times = val_times
        self.test_times = test_times
        self.features_list = features_list
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.normalize_data = normalize_data
        self.normalize_label = normalize_label
        self.pred_batch_size = pred_batch_size
        #self.label_weights = label_weights
        
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def train_dataloader(self):
        train_ds = TrainDataset(
            #n_prev = self.n_previous_times,
            first_lag = self.time_0,
            last_lag = self.time_0 + self.train_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label,
            #label_bins = self.label_bins,
            #label_weights = self.train_label_weights
        )
        return DataLoader(
            train_ds,
            batch_size = self.train_batch_size,
            shuffle = True,
            #sampler=WeightedRandomSampler(self.train_weights, len(train_ds)),
            num_workers = self.train_num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        val_ds = TrainDataset(
            #n_prev = self.n_previous_times,
            first_lag = self.time_0 + self.train_times,
            last_lag = self.time_0 + self.train_times + self.val_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label,
            #label_bins = self.label_bins,
            #label_weights = self.train_label_weights
        )
        return DataLoader(
            val_ds,
            batch_size = self.train_batch_size,
            #sampler=WeightedRandomSampler(self.val_weights, len(val_ds)),
            persistent_workers=True,
            num_workers = self.train_num_workers
        )
    
    def predict_dataloader(self):
        self.prediction_ds = PredictionDataset(
            #n_prev = self.n_previous_times,
            first_lag = self.time_0,
            last_lag = self.time_0 + self.train_times + self.val_times + self.test_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label
        )
        return DataLoader(
            self.prediction_ds,
            batch_size = self.pred_batch_size,
            #num_workers = self.train_num_workers
        )
    


class TrainDataset(Dataset):
    def __init__(self, first_lag, last_lag, features_list, normalize_data, normalize_label):
        #self.n_prev = n_prev
        self.first_lag = first_lag
        self.last_lag = last_lag
        self.features_list = features_list

        self.dataset = FeatureDataSet(features_list, masked=True, normalize_data=normalize_data, normalize_label=normalize_label)

    def __len__(self):
        return (self.last_lag - self.first_lag) * (self.dataset.n_vectors())
    
    def __getitem__(self, index):
        lag_i = (index // self.dataset.n_vectors()) + self.first_lag
        vector_i = index % self.dataset.n_vectors()
        
        data, label = self.dataset.get_data(lag_i, vector_i)
        
        if label == 0:
            label = np.array(0, dtype=np.int64)
            weight = 1
        else:
            label = np.array(1, dtype=np.int64)
            weight = 10
            
        return data, label, weight, lag_i, vector_i


class PredictionDataset(Dataset):
    def __init__(self, first_lag, last_lag, features_list, normalize_data, normalize_label):
        #self.n_prev = n_prev
        self.first_lag = first_lag
        self.last_lag = last_lag
        self.features_list = features_list
        
        self.mask = load_sb_image(mask_path).flatten()
        
        self.dataset = FeatureDataSet(features_list, masked=False, normalize_data=normalize_data, normalize_label=normalize_label)

    def __len__(self):
        return (self.last_lag - self.first_lag) * (self.dataset.n_vectors())
    
    def __getitem__(self, index):
        lag_i = (index // self.dataset.n_vectors()) + self.first_lag
        vector_i = index % self.dataset.n_vectors()
        
        mask_i = self.mask[vector_i]
        
        data, label = self.dataset.get_data(lag_i, vector_i)
        
        return data, label, mask_i, lag_i, vector_i