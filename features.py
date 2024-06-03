from pathlib import Path
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange, repeat
from enum import Enum
import numpy as np
from itertools import product

class PERIOD(Enum):
    BIWEEKLY = 0
    QUARTERLY = 1
    STATIC = 2

#PATHS
base_data_path = Path(r'/home/felferrari/projects/def_forecast/data/tiff/25k/v2')
mask_path = base_data_path / 'mask.tif'
# coordinates_shp_path = base_data_path / 'shp/coordinates.shp'

#features
features = {
    'ArDS':{
        'path_to_file' : base_data_path / 'ArDS.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'Biweekly':{
        'path_to_file' : base_data_path / 'Biweekly.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'AcAr':{
        'path_to_file' : base_data_path / 'AcAr.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'CtDS':{
        'path_to_file' : base_data_path / 'CtDS.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'DeAr':{
        'path_to_file' : base_data_path / 'DeAr.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 1
    },
    'Cloud':{
        'path_to_file' : base_data_path / 'nv.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 14
    },
    'OcDS':{
        'path_to_file' : base_data_path / 'OcDS.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'XQ':{
        'path_to_file' : base_data_path / 'XQ.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 4
    },
    'Flor':{
        'path_to_file' : base_data_path / 'Flor.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'A7Q':{
        'path_to_file' : base_data_path / 'A7Q.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 24
    },
    'NuAI':{
        'path_to_file' : base_data_path / 'NuAI.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'DeAI':{
        'path_to_file' : base_data_path / 'DeAI.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 1
    },
    'PtDG':{
        'path_to_file' : base_data_path / 'PtDG.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'PtEM':{
        'path_to_file' : base_data_path / 'PtEM.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'XArDS':{
        'path_to_file' : base_data_path / 'XArDS.tif',
        'period': PERIOD.QUARTERLY,
        'first_lag': 4 #in quarters
    },
    'XDeDS':{
        'path_to_file' : base_data_path / 'XDeDS.tif',
        'period': PERIOD.QUARTERLY,
        'first_lag': 4 #in quarters
    },
    'DS':{
        'path_to_file' : base_data_path / 'DS.tif',
        'period': PERIOD.QUARTERLY,
        'first_lag': 0 #in quarters
    },
    'Accessability':{
        'path_to_file' : base_data_path / 'Accessability.tif',
        'period': PERIOD.STATIC
    },
    'DryMonths':{
        'path_to_file' : base_data_path / 'DryMonths.tif',
        'period': PERIOD.STATIC
    },
    'Coordinates':{
        'path_to_file' : base_data_path / 'Coordinates.tif',
        'period': PERIOD.STATIC
    },
    'Distbd':{
        'path_to_file' : base_data_path / 'Dist.tif',
        'period': PERIOD.STATIC
    },
    'Dvd':{
        'path_to_file' : base_data_path / 'Dvd.tif',
        'period': PERIOD.STATIC
    },
    'EF':{
        'path_to_file' : base_data_path / 'EF.tif',
        'period': PERIOD.STATIC
    },
}


class FeatureData:
    def __init__(self, feat) -> None:
        self.name = feat.split('_')[0]
        if features[self.name]['period'] != PERIOD.STATIC:
            self.first_data_lag = features[self.name]['first_lag']
            if len(feat.split('_')) == 1:
                self.n_prev = 1
            else:
                self.n_prev = int(feat.split('_')[1])
                
        self.data = rearrange(load_ml_image(features[self.name]['path_to_file']), 'h w l -> l (h w)')
            
        if features[self.name]['period'] == PERIOD.STATIC:
            assert len(feat.split('_')) == 2, 'Static data must specifiy the layer'
            s_data = feat.split('_')[1]
            l_data = s_data.split(',')
            n_data = [int(d) for d in l_data]
            self.data = self.data[n_data]
        else:
            self.first_lag, self.last_lag = None, None
        
        mask_flatten = load_sb_image(mask_path).flatten()
        self.max = self.data[:, mask_flatten== 1].max()
        self.min = self.data[:, mask_flatten== 1].min()
        self.std = self.data[:, mask_flatten== 1].std()
        self.mean = self.data[:, mask_flatten== 1].mean()
        
        self.period = features[self.name]['period']
        # self.filtered = False
        
    def normalize_data(self):
        #self.data = (self.data - self.min) / (self.max - self.min)
        self.data = (self.data - self.mean) / (self.std)
    
    # def filter_period(self, first_lag, last_lag):
    #     assert not self.filtered, 'DataFeature only can be filtered once.'
    #     self.filtered = True

    #     if self.period == PERIOD.BIWEEKLY:
    #         assert (first_lag - self.first_data_lag >= 0) and (first_lag - self.first_data_lag <= self.data.shape[0]), f'Lag must be into the {self.name} limits'
    #         assert (last_lag - self.first_data_lag >= 0) and (last_lag - self.first_data_lag <= self.data.shape[0]), f'Lag must be into the {self.name} limits'
    #         self.data = self.data[first_lag - self.first_data_lag:  last_lag - self.first_data_lag]
            

    def get_data(self, lag_i, vector_i):
        if self.period == PERIOD.BIWEEKLY:
            lag_0 = lag_i - self.n_prev - self.first_data_lag
            lag_l = lag_i - self.first_data_lag
            assert lag_0 >=0 and lag_l < self.data.shape[0], 'lag_0 >=0 and lag_l < self.data.shape[0]' + f'{self.name}'
            return self.data[lag_0:lag_l, vector_i]
        elif self.period == PERIOD.QUARTERLY:
            lag_0 = lag_i // 6 - self.n_prev - self.first_data_lag + 1
            lag_l = lag_i //6 - self.first_data_lag + 1
            assert lag_0 >=0 and lag_l <= self.data.shape[0]
            return self.data[lag_0:lag_l, vector_i]
        elif self.period == PERIOD.STATIC:
            return self.data[:, vector_i]
        


class FeatureDataSet():
    def __init__(self, feature_list, lag_0, lag_size, normalize_data = False, normalize_label = False, geo_mask = False, mask = False, label_weights_bins = None, sample_bins = None) -> None:
        self.label = FeatureData(feature_list[0])
        self.features = [FeatureData(feat) for feat in feature_list]
        self.mask_flatten = load_sb_image(mask_path).flatten()
        
        #self.indexes = np.arange(len(self.label.data.flatten()))
        #self.indexes = self.indexes.reshape(self.label.data.shape)
        
        self.masked_indexes = np.ones_like(self.label.data)
        self.masked_indexes[:lag_0] = 0
        self.masked_indexes[lag_0+lag_size:] = 0
        
        if geo_mask:
            self.masked_indexes[:, self.mask_flatten == 0] = 0
            
        if normalize_label:
            self.label.normalize_data(self.masked_indexes)
            
        if normalize_data:
            for feature in self.features:
                feature.normalize_data()
                
        self.indexes = np.array(list(product(
            np.arange(self.label.data.shape[0]),
            np.arange(self.label.data.shape[1])
        )))
        
        self.indexes = self.indexes[self.masked_indexes.flatten() == 1]
        
        self.label_weights = np.zeros_like(self.label.data)
        self.label_weights[self.masked_indexes == 1] = 1
        
        if label_weights_bins is not None:
            count = []
            cond = (self.masked_indexes == 1) & (self.label.data == label_weights_bins[0])
            count_0 = cond.sum()
            count.append(count_0)
            
            if len(label_weights_bins) > 1:
                for bin_i in range(1, len(label_weights_bins)):
                    cond = (self.masked_indexes == 1) & (self.label.data > label_weights_bins[bin_i-1]) & (self.label.data <= label_weights_bins[bin_i])
                    count_i = cond.sum()
                    count.append(count_i)
                    weight = count_0 / count_i
                    self.label_weights[cond] = weight
                    
            cond = (self.masked_indexes == 1) & (self.label.data > label_weights_bins[-1])
            count_i = cond.sum()
            count.append(count_i)
            weight = count_0 / count_i
            self.label_weights[cond] = weight
            
        if sample_bins is not None:
            label_weights_ = np.zeros_like(self.label.data)
            label_weights_[self.masked_indexes == 1] = 1
            
            count = []
            cond = (self.masked_indexes == 1) & (self.label.data == sample_bins[0])
            count_0 = cond.sum()
            count.append(count_0)
            
            if len(sample_bins) > 1:
                for bin_i in range(1, len(sample_bins)):
                    cond = (self.masked_indexes == 1) & (self.label.data > sample_bins[bin_i-1]) & (self.label.data <= sample_bins[bin_i])
                    count_i = cond.sum()
                    count.append(count_i)
                    weight = count_0 / count_i
                    label_weights_[cond] = weight
                    
            cond = (self.masked_indexes == 1) & (self.label.data > sample_bins[-1])
            count_i = cond.sum()
            count.append(count_i)
            weight = count_0 / count_i
            label_weights_[cond] = weight
            
            self.sample_weights = label_weights_[self.masked_indexes == 1]
                
    # @property
    # def valid_indexes(self):
    #     return self.indexes[self.masked_indexes == 1]
    
    def __len__(self):
        return self.indexes.shape[0]
    
    def get_data(self, index):
        lag_i, vector_i = self.indexes[index]
        label = np.array([self.label.data[lag_i, vector_i]])
        #weight = self.mask_flatten[vector_i]
        weight = self.label_weights[lag_i, vector_i]
        data = {}
        for feature in self.features:
            data[feature.name] = feature.get_data(lag_i, vector_i)
        return data, label, weight, lag_i, vector_i
        

