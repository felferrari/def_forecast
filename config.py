from models.models import Resunet, Mlp
from utils.image_datasets import ImageDataModule
from utils.vector_datasets import VectorDataModule
import torch
from utils.callbacks import SaveImagePrediction, SaveVectorPrediction
import models
import utils

class default:
    n_prev_times = 12
    n_train_times = 72
    n_val_times = 48
    n_test_times = 48
    patch_size = 32
    train_overlap = 0.8
    train_batch_size = 32
    train_num_workers = 1
    prediction_overlap = 0.5
    prediction_border_removal = 4
    features_list = ['ArDS']  # first element is the target feature

image_data_module = {
    'class': ImageDataModule,
    'params':{
        'n_previous_times' : default.n_prev_times,
        'train_times' : default.n_train_times,
        'val_times' : default.n_val_times,
        'test_times' : default.n_test_times,
        'patch_size' : default.patch_size,
        'train_overlap' : default.train_overlap,
        'train_batch_size' : default.train_batch_size,
        'train_num_workers' : default.train_num_workers,
        'predictio_overlap' : default.prediction_overlap,
        'features' : default.features_list
    }
}

vector_data_module = {
    'class': VectorDataModule,
    'params':{
        'n_previous_times' : default.n_prev_times,
        'train_times' : default.n_train_times,
        'val_times' : default.n_val_times,
        'test_times' : default.n_test_times,
        'train_batch_size' : default.train_batch_size,
        'train_num_workers' : default.train_num_workers,
        'features_list' : default.features_list,
        'normalize_data' : True,
        'normalize_label' : False
    }
}


image_save_pred_callback = {
    'class' : SaveImagePrediction,
    'params':{
        'n_prev' : default.n_prev_times,
        'patch_size' : default.patch_size,
        'test_times' : default.n_test_times,
        'border_removal' : default.prediction_border_removal,
        'log_tiff' : True
    }
}

vector_save_pred_callback = {
    'class' : SaveVectorPrediction,
    'params':{
        'n_prev' : default.n_prev_times,
        'test_times' : default.n_test_times,
        'log_tiff' : True
    }
}



experiments = {
    'base':{
        'experiment_name': 'Deforestation Prediction',
        
        'criterion': {
            'class' : torch.nn.MSELoss,
            'params':{
                'reduction': 'none'
            }
        },

        
        'train_params':{
            'patience': 10,
            'accelerator' : 'gpu',
            'limit_train_batches': 6000,
            'limit_val_batches': 3000,
            
        },
        'pred_params':{
            'accelerator' : 'gpu'
        }
    }
}

experiments['resunet_base'] = experiments['base']
experiments['resunet_base'].update({
    'run_name': 'resunet',
    'model_name': 'resunet',
    'model': {
        'class': Resunet,
        'params':{
            'in_depth' : default.n_prev_times,
            'depths': [32, 64, 128, 256],
        }            
    },
    'optimizer' : {
        'class' : torch.optim.Adam,
        'params':{
            'lr': 1e-6
        }
    },
    'data_module': image_data_module,
    'save_pred_callback': image_save_pred_callback,
})


experiments['resunet'] = experiments['resunet_base']

experiments['mlp_vector_base'] = experiments['base']
experiments['mlp_vector_base'].update({
    'run_name': 'mlp',
    'model_name': 'mlp',
    'model': {
        'class': Mlp,
        'params':{
            #'in_depth' : default.n_prev_times,
            'layers':[64, 256, 512, 1024, 512, 256, 128]
        }            
    },
    'optimizer' : {
        'class' : torch.optim.Adam,
        'params':{
            'lr': 1e-6
        }
    },
    'data_module': vector_data_module,
    'save_pred_callback': vector_save_pred_callback,
})
experiments['mlp_vector_base']['data_module']['params'].update({'train_batch_size' : 64})
experiments['mlp_vector_base']['data_module']['params'].update({'train_num_workers' : 12})

experiments['mlp'] = experiments['mlp_vector_base']