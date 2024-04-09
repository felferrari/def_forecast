from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
import numpy as np
from features import FeatureDataSet


class DataModule(LightningDataModule):
    def __init__(
        self,
        features_list,
        time_0 = 24,
        train_times = 48,
        val_times = 48,
        test_times = 48,
        train_batch_size = 128,
        test_batch_size = 256,
        train_num_workers = 8,
        test_num_workers = 8,
        pred_batch_size = 2048,
        normalize_data = True,
        normalize_label = False,
        task = None,
        label_weights_bins = None,
        sample_bins = None,
        *args, **kargs
        ) -> None:
        super().__init__()
        self.time_0 = time_0
        self.train_times = train_times
        self.val_times = val_times
        self.test_times = test_times
        self.features_list = features_list
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.test_num_workers = test_num_workers
        self.normalize_data = normalize_data
        self.normalize_label = normalize_label
        self.pred_batch_size = pred_batch_size
        self.task = task
        self.label_weights_bins = label_weights_bins
        self.sample_bins = sample_bins
        
        #data_labels = FeatureDataSet(features_list, lag_0 = time_0, lag_size = train_times+val_times, geo_mask=True)
        
    def train_dataloader(self):
        if self.task == 'classification':
            train_ds = VectorClassificationDataset(
                features_list=self.features_list,
                lag_0=self.time_0,
                lag_size=self.train_times,
                geo_mask=True,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                label_weights_bins = self.label_weights_bins,
                sample_bins = self.sample_bins
                )
        elif self.task == 'regression':
            train_ds = VectorRegressionDataset(
                features_list=self.features_list,
                lag_0=self.time_0,
                lag_size=self.train_times,
                geo_mask=True,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                label_weights_bins = self.label_weights_bins,
                sample_bins = self.sample_bins
                )
        #train_ds = Subset(train_ds, train_ds.feature_dataset.valid_indexes)
        
        if self.sample_bins is not None:
            sampler = WeightedRandomSampler(
                weights = train_ds.feature_dataset.sample_weights,
                num_samples = len(train_ds.feature_dataset.sample_weights)
            )
        else:
            sampler = RandomSampler(
                data_source=train_ds
            )
        
        train_dl = DataLoader(
            train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            persistent_workers=True,
            sampler=sampler,
            pin_memory=True
            )
        
        return train_dl
        
    def val_dataloader(self):

        if self.task == 'classification':
            val_ds = VectorClassificationDataset(
                features_list=self.features_list,
                lag_0=self.time_0 + self.train_times,
                lag_size=self.val_times,
                geo_mask=True,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                label_weights_bins = self.label_weights_bins,
                sample_bins = self.sample_bins
                )
        elif self.task == 'regression':
            val_ds = VectorRegressionDataset(
                features_list=self.features_list,
                lag_0=self.time_0 + self.train_times,
                lag_size=self.val_times,
                geo_mask=True,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                label_weights_bins = self.label_weights_bins,
                sample_bins = self.sample_bins
                )
            
        if self.sample_bins is not None:
            sampler = WeightedRandomSampler(
                val_ds.feature_dataset.sample_weights,
                num_samples = len(val_ds.feature_dataset.sample_weights)
            )
        else:
            sampler = None
        
        #val_ds = Subset(val_ds, val_ds.feature_dataset.valid_indexes)
        
        val_dl = DataLoader(
            val_ds,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            persistent_workers=True,
            sampler = sampler,
            pin_memory=True
            )
        
        return val_dl
    
    def predict_dataloader(self):

        if self.task == 'classification':
            self.predict_ds = VectorClassificationDataset(
                features_list=self.features_list,
                lag_0=self.time_0 + self.train_times + self.val_times,
                lag_size=self.test_times,
                geo_mask=False,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                )
        elif self.task == 'regression':
            self.predict_ds = VectorRegressionDataset(
                features_list=self.features_list,
                lag_0=self.time_0 + self.train_times + self.val_times,
                lag_size=self.test_times,
                geo_mask=False,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                )
        
        #val_ds = Subset(val_ds, val_ds.feature_dataset.valid_indexes)
        
        pred_dl = DataLoader(
            self.predict_ds,
            batch_size=self.pred_batch_size,
            num_workers=1,
            pin_memory=True
            )
        
        return pred_dl
    
    def test_dataloader(self):

        if self.task == 'classification':
            val_ds = VectorClassificationDataset(
                features_list=self.features_list,
                lag_0=self.time_0 + self.train_times + self.val_times,
                lag_size=self.test_times,
                geo_mask=True,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                )
        elif self.task == 'regression':
            val_ds = VectorRegressionDataset(
                features_list=self.features_list,
                lag_0=self.time_0 + self.train_times + self.val_times,
                lag_size=self.test_times,
                geo_mask=True,
                normalize_data =  self.normalize_data,
                normalize_label = self.normalize_label,
                )
            
        #val_ds = Subset(val_ds, val_ds.feature_dataset.valid_indexes)
        
        val_dl = DataLoader(
            val_ds,
            batch_size=self.test_batch_size,
            num_workers=self.test_num_workers,
            #persistent_workers=True,
            #sampler = sampler,
            shuffle=True,
            pin_memory=True
            )
        
        return val_dl
    
class VectorDataset(Dataset):
    def __init__(self,
                 features_list, 
                 lag_0, 
                 lag_size, 
                 geo_mask,
                 normalize_data, 
                 normalize_label,
                 label_weights_bins = None,
                 sample_bins = None
                 ) -> None:
        
        self.feature_dataset = FeatureDataSet(
            features_list, 
            lag_0 = lag_0, 
            lag_size = lag_size, 
            geo_mask = geo_mask, 
            normalize_data = normalize_data,
            normalize_label = normalize_label,
            label_weights_bins = label_weights_bins, 
            sample_bins = sample_bins)
        
    def __len__(self):
        return len(self.feature_dataset)

class VectorClassificationDataset(VectorDataset):
    
    def __getitem__(self, index):
        data, label, weight, lag_i, vector_i = self.feature_dataset.get_data(index)
        
        if label == 0:
            label = np.array([1., 0.]).astype(np.float32)
        else:
            label = np.array([0., 1.]).astype(np.float32)
            
        #weight = np.array([weight, weight]).astype(np.float32)
        
        return data, label, weight, lag_i, vector_i
    
class VectorRegressionDataset(VectorDataset):
    
    def __getitem__(self, index):
        data, label, weight, lag_i, vector_i = self.feature_dataset.get_data(index)
        
        #weight = np.array([weight, weight]).astype(np.float32)
        
        return data, label, weight, lag_i, vector_i