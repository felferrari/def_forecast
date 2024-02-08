from torch.utils.data import Dataset
import fiona
import numpy as np
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange
from skimage.util import view_as_windows

class TrainDataset(Dataset):
    def __init__(self, data_path, mask_path, n_prev, bands, patch_size = 32, overlap = 0.7):
        self.patch_size = patch_size
        self.n_prev = n_prev
        self.bands = bands

        data = load_ml_image(data_path)
        mask = load_sb_image(mask_path)
        shape = mask.shape
        self.mask = mask.flatten()

        self.data = rearrange(data, 'h w c-> (h w) c')
        idx_array = rearrange(np.arange(shape[0]*shape[1]), f' (h w) -> h w', h = shape[0], w = shape[1])
        self.patches_idx = rearrange(view_as_windows(idx_array, (patch_size, patch_size), int((1-overlap)*patch_size)), 'nh nw h w -> (nh nw) h w')

        #clean patches outside the mask
        self.patches_idx = self.patches_idx[np.any((self.mask[self.patches_idx] == 1), axis=(1,2))]


    def __len__(self):
        return self.patches_idx.shape[0]*((self.bands[1] - self.bands[0]) - self.n_prev)
    
    def __getitem__(self, index):
        patch_i = index % self.patches_idx.shape[0]
        band_date_i = index // self.patches_idx.shape[0] + self.n_prev

        patch_idx = self.patches_idx[patch_i]

        mask = self.mask[patch_idx]

        ref = self.data[patch_idx][:,:,band_date_i]

        data = self.data[patch_idx][:,:,band_date_i-self.n_prev: band_date_i]

        data = rearrange(data, 'h w c -> c h w')
        ref = rearrange(ref, 'h w -> 1 h w')
        mask = rearrange(mask, 'h w -> 1 h w')


        return data, ref, mask

        

