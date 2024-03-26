from pathlib import Path
from utils.ops import load_ml_image, load_sb_image
from einops import rearrange
from enum import Enum
import numpy as np

class PERIOD(Enum):
    BIWEEKLY = 0
    QUARTERLY = 1
    STATIC = 2

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
    'Biweekly':{
        'path_to_file' : base_data_path / 'tiff/Biweekly.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'AcAr':{
        'path_to_file' : base_data_path / 'tiff/AcAr.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'CtDS':{
        'path_to_file' : base_data_path / 'tiff/CtDS.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'DeAr':{
        'path_to_file' : base_data_path / 'tiff/DeAr.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 1
    },
    'Cloud':{
        'path_to_file' : base_data_path / 'tiff/nv.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 14
    },
    'OcDS':{
        'path_to_file' : base_data_path / 'tiff/OcDS.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 0
    },
    'XQ':{
        'path_to_file' : base_data_path / 'tiff/XQ.tif',
        'period': PERIOD.BIWEEKLY,
        'first_lag': 4
    },
    'XArDS':{
        'path_to_file' : base_data_path / 'tiff/XArDS.tif',
        'period': PERIOD.QUARTERLY,
        'first_lag': 4 #in quarters
    },
    'XDeDS':{
        'path_to_file' : base_data_path / 'tiff/XDeDS.tif',
        'period': PERIOD.QUARTERLY,
        'first_lag': 4 #in quarters
    },
    'DS':{
        'path_to_file' : base_data_path / 'tiff/DS.tif',
        'period': PERIOD.QUARTERLY,
        'first_lag': 0 #in quarters
    },
    'DryMonths':{
        'path_to_file' : base_data_path / 'tiff/DryMonths_bd_amz_25km.tif',
        'period': PERIOD.STATIC
    },
    'Coordinates':{
        'path_to_file' : base_data_path / 'tiff/Coordinates.tif',
        'period': PERIOD.STATIC
    },
    'Distbd':{
        'path_to_file' : base_data_path / 'tiff/Dist_bd_amz_25km.tif',
        'period': PERIOD.STATIC
    },
    'Dvd':{
        'path_to_file' : base_data_path / 'tiff/Dvd_bd_amz_25km.tif',
        'period': PERIOD.STATIC
    },
    'EF':{
        'path_to_file' : base_data_path / 'tiff/EF_bd_amz_25km.tif',
        'period': PERIOD.STATIC
    }
}


class FeatureData:
    def __init__(self, feat, masked) -> None:
        self.name = feat.split('_')[0]
        if features[self.name]['period'] != PERIOD.STATIC:
            self.first_data_lag = features[self.name]['first_lag']
            if len(feat.split('_')) == 1:
                self.n_prev = 1
            else:
                self.n_prev = int(feat.split('_')[1])
                
        self.data = rearrange(load_ml_image(features[self.name]['path_to_file']), 'h w l -> (h w) l')
        if masked:
            mask = load_sb_image(mask_path).flatten()
            self.data = self.data[mask == 1]
            
        self.data = rearrange(self.data, 'n l ->l n')
        
        if features[self.name]['period'] == PERIOD.STATIC:
            assert len(feat.split('_')) == 2, 'Static data must specifiy the layer'
            s_data = feat.split('_')[1]
            l_data = s_data.split(',')
            n_data = [int(d) for d in l_data]
            self.data = self.data[n_data]
        else:
            self.first_lag, self.last_lag = None, None
        
        self.max = self.data.max()
        self.min = self.data.min()
        self.std = self.data.std()
        self.mean = self.data.mean()
        
        self.period = features[self.name]['period']
        self.filtered = False
        
    def normalize_data(self):
        self.data = (self.data - self.min) / (self.max - self.min)
    
    def filter_period(self, first_lag, last_lag):
        assert not self.filtered, 'DataFeature only can be filtered once.'
        self.filtered = True
        if self.period == PERIOD.STATIC or self.period == PERIOD.QUARTERLY:
            return
        
        if self.period == PERIOD.BIWEEKLY:
            assert (first_lag - self.first_data_lag >= 0) and (first_lag - self.first_data_lag <= self.data.shape[0]), f'Lag must be into the {self.name} limits'
            assert (last_lag - self.first_data_lag >= 0) and (last_lag - self.first_data_lag <= self.data.shape[0]), f'Lag must be into the {self.name} limits'
            self.data = self.data[first_lag - self.first_data_lag:  last_lag - self.first_data_lag]
            
    def get_data(self, lag_i, vector_i):
        if self.period == PERIOD.BIWEEKLY:
            lag_0 = lag_i - self.n_prev - self.first_data_lag
            lag_l = lag_i - self.first_data_lag
            assert lag_0 >=0 and lag_l < self.data.shape[0]
            return self.data[lag_0:lag_l, vector_i]
        elif self.period == PERIOD.QUARTERLY:
            lag_0 = lag_i // 6 - self.n_prev - self.first_data_lag + 1
            lag_l = lag_i //6 - self.first_data_lag + 1
            assert lag_0 >=0 and lag_l <= self.data.shape[0]
            return self.data[lag_0:lag_l, vector_i]
        elif self.period == PERIOD.STATIC:
            return self.data[:, vector_i]
        
class FeatureDataSet():
    def __init__(self, feature_list, normalize_data = False, normalize_label = False, masked = False) -> None:
        self.label = FeatureData(feature_list[0], masked)
        self.features = [FeatureData(feat, masked) for feat in feature_list]
        #self.first_dataset_lag = get_first_lag(feature_list)
        
        self.n_prev = 1
        for feat in self.features:
            if feat.period == PERIOD.STATIC:
                continue
            self.n_prev = max(self.n_prev, feat.n_prev)
        
        if normalize_label:
            self.label.normalize_data()
            
        if normalize_data:
            for feature in self.features:
                feature.normalize_data()
                
        self.first_lag = 0
        self.last_lag = self.label.data.shape[0]

    def n_lags(self):
        return self.label.data.shape[0]
    
    def n_vectors(self):
        return self.label.data.shape[1]
    
    def get_data(self, lag_i, vector_i):
        label = np.array([self.label.data[lag_i, vector_i]])
        data = {}
        for feature in self.features:
            data[feature.name] = feature.get_data(lag_i, vector_i)
        return data, label
        
