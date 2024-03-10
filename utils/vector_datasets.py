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
                 pred_batch_size,
                 sample_bins,
                 label_bins,
                 label_weights,
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
        self.pred_batch_size = pred_batch_size
        self.label_bins = label_bins
        self.label_weights = label_weights
        
        
        #mask = load_sb_image(mask_path).flatten()
        
        #deforestation data
        self.first_lag = get_first_lag(features_list)
               
        #train_data = def_data.data[n_previous_times + self.first_lag : train_times]
        train_data = FeatureData(features_list[0], masked=True)
        train_data.filter_period(n_previous_times + self.first_lag, train_times)
        #val_data = def_data.data[n_previous_times + train_times : train_times + val_times]
        val_data = FeatureData(features_list[0], masked=True)
        val_data.filter_period(n_previous_times + train_times, train_times + val_times)
        
        self.train_weights = np.ones_like(train_data.data.flatten())
        self.val_weights = np.ones_like(val_data.data.flatten())

        if sample_bins is not None: 
            train_n_bin_0 = (train_data.data == 0).sum() 
            val_n_bin_0 = (val_data.data == 0).sum() 
            n_bins = len(sample_bins)
            if n_bins > 1:
                for i in range(1, n_bins):
                    train_conditions = np.logical_and(train_data.data > sample_bins[i-1], train_data.data <= sample_bins[i])
                    val_conditions = np.logical_and(val_data.data > sample_bins[i-1], val_data.data <= sample_bins[i])
                    train_conditions = rearrange(train_conditions, 'l n -> (l n)')
                    val_conditions = rearrange(val_conditions, 'l n -> (l n)')
                    
                    train_prop = train_n_bin_0 / train_conditions.sum()
                    val_prop = val_n_bin_0 / val_conditions.sum()
                    
                    self.train_weights[train_conditions] = train_prop
                    self.val_weights[val_conditions] = val_prop
                    
            train_conditions = train_data.data > sample_bins[-1]
            val_conditions = val_data.data > sample_bins[-1]
            train_conditions = rearrange(train_conditions, 'l n -> (l n)')
            val_conditions = rearrange(val_conditions, 'l n -> (l n)')
            
            train_prop = train_n_bin_0 / train_conditions.sum()
            val_prop = val_n_bin_0 / val_conditions.sum()
            
            self.train_weights[train_conditions] = train_prop
            self.val_weights[val_conditions] = val_prop
            

        
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def train_dataloader(self):
        train_ds = VectorTrainDataset(
            n_prev = self.n_previous_times,
            first_lag = self.first_lag,
            last_lag = self.train_times,
            features_list = self.features_list,
            normalize_data=self.normalize_data,
            normalize_label=self.normalize_label,
            label_bins = self.label_bins,
            label_weights = self.label_weights
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
            normalize_label=self.normalize_label,
            label_bins = self.label_bins,
            label_weights = self.label_weights
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
            normalize_label=self.normalize_label,
            label_bins = self.label_bins,
            label_weights = self.label_weights
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
            batch_size = self.pred_batch_size,
            #num_workers = self.train_num_workers
        )
    


class VectorTrainDataset(Dataset):
    def __init__(self, n_prev, first_lag, last_lag, features_list, normalize_data, normalize_label, label_bins, label_weights):
        self.n_prev = n_prev
        self.first_lag = first_lag
        self.last_lag = last_lag
        self.features_list = features_list
        self.label_bins = label_bins
        self.label_weights = label_weights
        
        assert ((label_bins is None) and (label_weights is None)) or len(label_bins) == len(label_weights) - 1, 'Label Weights ans bins must be compatible'
        
        self.dataset = FeatureDataSet(features_list, masked=True, normalize_data=normalize_data, normalize_label=normalize_label)
        self.dataset.filter_period(first_lag, last_lag)

    def __len__(self):
        return (self.dataset.n_lags() - self.n_prev) * (self.dataset.n_vectors())
    
    def __getitem__(self, index):
        lag_i = (index // self.dataset.n_vectors()) + self.n_prev
        vector_i = index % self.dataset.n_vectors()
        
        data, label = self.dataset.get_data(lag_i, vector_i, self.n_prev)
        
        if self.label_bins is not None:
            if label == self.label_bins[0]:
                weight = self.label_weights[0]
            else:
                if len(self.label_bins) > 1:
                    for i in range(1, len(self.label_bins)):
                        if label > self.label_bins[i-1] and label <= self.label_bins[i]:
                            weight = self.label_weights[i]
                if label > self.label_bins[-1]:
                    weight = self.label_weights[-1]
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