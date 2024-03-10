from pathlib import Path
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange, repeat
from enum import Enum
import numpy as np

class PERIOD(Enum):
    BIWEEKLY = 0
    QUARTERLY = 1
    FIXED = 2

#PATHS
base_data_path = Path(r'/home/felferrari/projects/def_forecast/data')
mask_path = base_data_path / r'tiff/mask.tif'

#features
features = {
    'ArDS':{
        'path_to_file' : base_data_path / 'tiff/ArDS.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'DeAr':{
        'path_to_file' : base_data_path / 'tiff/DeAr.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 1
    },
    'XArDS':{
        'path_to_file' : base_data_path / 'tiff/XArDS.tif',
        'period': PERIOD.QUARTERLY,
        'first_lag': 24
    },
    'DryMonths':{
        'path_to_file' : base_data_path / 'tiff/DryMonths_bd_amz_25km.tif',
        'period': PERIOD.FIXED
    },
    'Coordinates':{
        'path_to_file' : base_data_path / 'tiff/Coordinates.tif',
        'period': PERIOD.FIXED
    },
}

class FeatureData:
    def __init__(self, feat, masked) -> None:
        self.name = feat
        self.feature = features[feat]
        mask = load_sb_image(mask_path).flatten()
        self.data = rearrange(load_ml_image(self.feature['path_to_file']), 'h w l -> (h w) l')
        if masked:
            self.data = self.data[mask == 1]
        self.data = rearrange(self.data, 'n l ->l n')
        self.max = self.data.max()
        self.min = self.data.min()
        self.period = features[feat]['period']
        if features[feat]['period'] == PERIOD.QUARTERLY:
            self.data = repeat(self.data, 'l n -> (repeat l) n', repeat = 6)
        if features[feat]['period'] != PERIOD.FIXED:
            self.first_data_lag = features[feat]['first_lag']
        
    def normalize_data(self):
        self.data = (self.data - self.min) / (self.max - self.min)
    
    def filter_period(self, fisrt_lag, last_lag):
        if self.period == PERIOD.FIXED:
            return
        assert (fisrt_lag - self.first_data_lag >= 0) and (fisrt_lag - self.first_data_lag <= self.data.shape[0]), f'Lag must be into the {self.name} limits'
        assert (last_lag - self.first_data_lag >= 0) and (last_lag - self.first_data_lag <= self.data.shape[0]), f'Lag must be into the {self.name} limits'
        self.data = self.data[fisrt_lag - self.first_data_lag:  last_lag - self.first_data_lag]


class FeatureDataSet():
    def __init__(self, feature_list, normalize_data = False, normalize_label = False, masked = False) -> None:
        self.label = FeatureData(feature_list[0], masked)
        self.features = [FeatureData(feat, masked) for feat in feature_list]
        self.first_dataset_lag = get_first_lag(feature_list)
        
        if normalize_label:
            self.label.normalize_data()
            
        if normalize_data:
            for feature in self.features:
                feature.normalize_data()
        
        
    def filter_period(self, first_lag, last_lag):
        self.label.filter_period(first_lag, last_lag)
        for feature in self.features:
            if feature.period == PERIOD.FIXED:
                continue
            feature.filter_period(first_lag, last_lag)
            assert  feature.data.shape[0] == self.label.data.shape[0], f'All features must have the same size. {feature.name} has a different shape[0].'
            assert  feature.data.shape[1] == self.label.data.shape[1], f'All features must have the same size. {feature.name} has a different shape[1].'
            
    def n_lags(self):
        return self.label.data.shape[0]
    
    def n_vectors(self):
        return self.label.data.shape[1]
    
    def get_data(self, lag_i, vector_i, n_prev):
        label = np.array([self.label.data[lag_i, vector_i]])
        data = {}
        for feature in self.features:
            if feature.period == PERIOD.FIXED:
                data[feature.name] = feature.data[:, vector_i]
            elif feature.period == PERIOD.QUARTERLY:
                data[feature.name] = feature.data[lag_i -1: lag_i, vector_i]
            elif feature.period == PERIOD.BIWEEKLY:
                data[feature.name] = feature.data[lag_i - n_prev: lag_i, vector_i]
                
        return data, label
        
        
        
        
    
        
def get_first_lag(feat_list):
    first_lag = 0
    for feat in feat_list:
        if features[feat]['period'] == PERIOD.FIXED:
            continue
        first_lag = max(first_lag, features[feat]['first_lag'])
    
    return first_lag

