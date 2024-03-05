from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import fiona
import numpy as np
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange
from skimage.util import view_as_windows
from lightning import LightningDataModule
import paths
import albumentations as A
from albumentations.augmentations.geometric.transforms import HorizontalFlip, VerticalFlip 

class ImageDataModule(LightningDataModule):
    def __init__(self,
                 n_previous_times,
                 train_times,
                 val_times,
                 test_times,
                 patch_size,
                 train_overlap,
                 train_batch_size,
                 train_num_workers,
                 prediction_overlap,
                 features
                 ) -> None:
        super().__init__()
        self.n_previous_times = n_previous_times
        self.train_times = train_times
        self.val_times = val_times
        self.test_times = test_times
        self.patch_size = patch_size
        self.train_overlap = train_overlap
        self.features = features
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.prediction_overlap = prediction_overlap
        

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def train_dataloader(self):
        train_ds = PatchTrainDataset(
            n_prev = self.n_previous_times,
            band_first = 0,
            band_last = self.train_times,
            features = self.features,
            patch_size = self.patch_size,
            prediction_overlap = self.train_overlap,
            mode='train'
        )
        return DataLoader(
            train_ds,
            batch_size = self.train_batch_size,
            shuffle = True,
            num_workers = self.train_num_workers
        )
    
    def val_dataloader(self):
        val_ds = PatchTrainDataset(
            n_prev = self.n_previous_times,
            band_first = self.train_times,
            band_last = self.train_times + self.val_times,
            features = self.features,
            patch_size = self.patch_size,
            prediction_overlap = self.train_overlap,
            mode='validation'
        )
        return DataLoader(
            val_ds,
            batch_size = self.train_batch_size,
            num_workers = self.train_num_workers
        )
    
    def test_dataloader(self):
        test_ds = PatchTrainDataset(
            n_prev = self.n_previous_times,
            band_first = self.train_times + self.val_times,
            band_last = self.train_times + self.val_times + self.test_times,
            features = self.features,
            patch_size = self.patch_size,
            prediction_overlap = self.train_overlap,
            mode='test'
        )
        return DataLoader(
            test_ds,
            batch_size = self.train_batch_size,
            num_workers = self.train_num_workers
        )
    
    def predict_dataloader(self):
        prediction_ds = PatchPredictionDataset(
            n_prev = self.n_previous_times,
            band_first = self.train_times + self.val_times,
            band_last = self.train_times + self.val_times + self.test_times,
            features = self.features,
            patch_size = self.patch_size,
            overlap = self.prediction_overlap
        )
        return DataLoader(
            prediction_ds,
            batch_size = self.train_batch_size,
            num_workers = self.train_num_workers
        )
    


class PatchTrainDataset(Dataset):
    def __init__(self, n_prev, band_first, band_last, features, mode, prediction_overlap, patch_size):
        self.patch_size = patch_size
        self.n_prev = n_prev
        self.band_first = band_first
        self.band_last = band_last
        self.mode = mode
        
        #deforestation data
        qd_paths_dict = paths.path_to_data['def_data']
        def_data = load_ml_image(qd_paths_dict)[:,:,band_first :band_last]
        self.def_data = rearrange(def_data, 'h w c -> (h w) c')
        
        
        mask = load_sb_image(paths.path_to_mask)
        shape = mask.shape
        self.mask = mask.flatten()

        idx_array = rearrange(np.arange(shape[0]*shape[1]), f' (h w) -> h w', h = shape[0], w = shape[1])
        self.patches_idx = rearrange(view_as_windows(idx_array, (patch_size, patch_size), int((1-prediction_overlap)*patch_size)), 'nh nw h w -> (nh nw) h w')

        #clean patches outside the mask
        self.patches_idx = self.patches_idx[np.any((self.mask[self.patches_idx] == 1), axis=(1,2))]
        
        if mode == 'train':
            self.augmentation_transform = A.Compose(
                [
                    VerticalFlip(),
                    HorizontalFlip()
                ],
                additional_targets = {
                    'ref': 'mask',
                    'mask1': 'mask'
                }
            )


    def __len__(self):
        return self.patches_idx.shape[0]*(self.band_last - self.band_first - self.n_prev)
    
    def __getitem__(self, index):
        patch_i = index % self.patches_idx.shape[0]
        band_date_i = index // self.patches_idx.shape[0] + self.n_prev

        patch_idx = self.patches_idx[patch_i]

        mask = self.mask[patch_idx]
        
        ref = self.def_data[patch_idx][:,:,band_date_i]

        data = self.def_data[patch_idx][:,:,band_date_i-self.n_prev: band_date_i] / 100
        
        if self.mode=='train':
            transformed = self.augmentation_transform(image=data, ref = ref, mask1 = mask)
            data = transformed['image']
            ref = transformed['ref']
            mask = transformed['mask1']
            
        data = rearrange(data, 'h w c -> c h w')
        ref = rearrange(ref, 'h w -> 1 h w')
        mask = rearrange(mask, 'h w -> 1 h w')
        
        if self.mode in ['train', 'validation']:
            #mask[np.logical_and(ref > 0, ref<=1)] = 5 * mask[np.logical_and(ref > 0, ref<=1)]
            #mask[np.logical_and(ref > 1, ref<=10)] = 10 * mask[np.logical_and(ref > 1, ref<=10)]
            #mask[ref > 10] = 100 * mask[ref > 10]
            #mask[ref > 0] = 100 * mask[ref > 0]
            pass
        
        return data, ref, mask


class PatchPredictionDataset(Dataset):
    def __init__(self, n_prev, band_first, band_last, features, patch_size, overlap):
        self.patch_size = patch_size
        self.n_prev = n_prev
        self.band_first = band_first
        self.band_last = band_last
        
        mask = load_sb_image(paths.path_to_mask)
        self.original_shape = mask.shape
        pad_width = ((patch_size, patch_size), (patch_size, patch_size))
        self.padded_mask = np.pad(mask, pad_width, 'reflect')
        self.padded_shape = self.padded_mask.shape
        self.padded_mask = self.padded_mask.flatten()
        
        #deforestation data
        qd_paths_dict = paths.path_to_data['def_data']
        def_data = load_ml_image(qd_paths_dict)[:,:,band_first:band_last]
        pad_width = ((patch_size, patch_size), (patch_size, patch_size), (0, 0))
        padded_def_data = np.pad(def_data, pad_width, 'reflect')
        self.padded_def_data = rearrange(padded_def_data, 'h w c -> (h w) c')

        #patches indexes
        idx_array = rearrange(np.arange(self.padded_shape[0]*self.padded_shape[1]), f' (h w) -> h w', h = self.padded_shape[0], w = self.padded_shape[1])
        self.patches_idx = rearrange(view_as_windows(idx_array, (patch_size, patch_size), int((1-overlap)*patch_size)), 'nh nw h w -> (nh nw) h w')

        #clean patches outside the mask
        self.patches_idx = self.patches_idx[np.any((self.padded_mask[self.patches_idx] == 1), axis=(1,2))]



    def __len__(self):
        return self.patches_idx.shape[0]*(self.band_last - self.band_first - self.n_prev)
    
    def __getitem__(self, index):
        patch_i = index % self.patches_idx.shape[0]
        band_date_i = index // self.patches_idx.shape[0] + self.n_prev

        patch_idx = self.patches_idx[patch_i]

        mask = self.padded_mask[patch_idx]

        ref = self.padded_def_data[patch_idx][:,:,band_date_i]

        data = self.padded_def_data[patch_idx][:,:,band_date_i-self.n_prev: band_date_i] / 100
        
           
        data = rearrange(data, 'h w c -> c h w')
        ref = rearrange(ref, 'h w -> 1 h w')
        mask = rearrange(mask, 'h w -> 1 h w')
        
        return data, ref, mask, patch_idx, band_date_i