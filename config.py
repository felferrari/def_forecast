from models.models import Resunet, Mlp
from utils.image_datasets import ImageDataModule
from utils.vector_datasets import VectorDataModule
import torch
from utils.callbacks import SaveImagePrediction, SaveVectorPrediction
import models
import utils
from copy import deepcopy

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
    pred_batch_size = 32
    train_sample_bins = None
    label_bins = None
    label_weights = None
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
        'prediction_overlap' : default.prediction_overlap,
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
        'pred_batch_size' : default.pred_batch_size,
        'sample_bins' : default.train_sample_bins,
        'label_bins' : default.label_bins,
        #'label_weights' : default.label_weights,
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
            'limit_train_batches': 1000,
            'limit_val_batches': None,
            
        },
        'pred_params':{
            'accelerator' : 'gpu'
        }
    }
}

experiments['resunet_base'] = deepcopy(experiments['base'])
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


experiments['resunet'] = deepcopy(experiments['resunet_base'])

experiments['mlp_vector_base'] = deepcopy(experiments['base'])
experiments['mlp_vector_base'].update({
    'run_name': 'mlp',
    'model_name': 'mlp',
    'model': {
        'class': Mlp,
        'params':{
            #'in_depth' : default.n_prev_times,
            'layers':[64, 256, 512, 256, 128]
        }            
    },
    'optimizer' : {
        'class' : torch.optim.Adam,
        'params':{
            'lr': 2e-7
        }
    },
    'data_module': vector_data_module,
    'save_pred_callback': vector_save_pred_callback,
})
experiments['mlp_vector_base']['data_module']['params'].update({'train_batch_size' : 128})
experiments['mlp_vector_base']['data_module']['params'].update({'pred_batch_size' : 2048})
experiments['mlp_vector_base']['data_module']['params'].update({'train_num_workers' : 4})

experiments['mlp'] = deepcopy(experiments['mlp_vector_base'])

experiments['zeros'] = deepcopy(experiments['mlp'])
experiments['zeros']['run_name'] = 'zeros'

#base Experiment
experiments['mlp_0'] = deepcopy(experiments['mlp'])
experiments['mlp_0']['run_name'] = 'mlp_0'

#Reweighting
experiments['mlp_1'] = deepcopy(experiments['mlp'])
experiments['mlp_1']['run_name'] = 'mlp_1'
experiments['mlp_1']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]

#Resampling
experiments['mlp_2'] = deepcopy(experiments['mlp'])
experiments['mlp_2']['run_name'] = 'mlp_2'
experiments['mlp_2']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]

#Reweighting and Resampling
experiments['mlp_3'] = deepcopy(experiments['mlp'])
experiments['mlp_3']['run_name'] = 'mlp_3'
experiments['mlp_3']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]
experiments['mlp_3']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]
# train_samples_cond = [
#     None,
#     [0],
#     [0, 1],
#     [0, 5],
#     [0, 10],
#     [0, 20],
#     [0, 1, 10],
#     [0, 1, 20],
#     [0, 5, 10],
#     [0, 5, 20],
#     [0, 10, 20],
#     [0, 1, 5, 10],
#     [0, 1, 5, 20]
# ]

# train_bins_weights = [
#     {
#         'bins': None,
#         'weights': None
#     },
#     {
#         'bins': [0],
#         'weights': [1, 10]
#     },
#     {
#         'bins': [0],
#         'weights': [1, 100]
#     },
#     {
#         'bins': [0],
#         'weights': [1, 1000]
#     },
#     {
#         'bins': [0],
#         'weights': [1, 10000]
#     },
#     {
#         'bins': [0, 1],
#         'weights': [1, 10, 100]
#     },
#     {
#         'bins': [0, 1],
#         'weights': [1, 10, 1000]
#     },
#     {
#         'bins': [0, 1],
#         'weights': [1, 100, 1000]
#     },
#     {
#         'bins': [0, 1],
#         'weights': [1, 100, 10000]
#     },
#     {
#         'bins': [0, 1],
#         'weights': [1, 1000, 10000]
#     },
#     {
#         'bins': [0, 5],
#         'weights': [1, 10, 100]
#     },
#     {
#         'bins': [0, 5],
#         'weights': [1, 10, 1000]
#     },
#     {
#         'bins': [0, 5],
#         'weights': [1, 100, 1000]
#     },
#     {
#         'bins': [0, 5],
#         'weights': [1, 100, 10000]
#     },
#     {
#         'bins': [0, 5],
#         'weights': [1, 1000, 10000]
#     },
#     {
#         'bins': [0, 10],
#         'weights': [1, 10, 100]
#     },
#     {
#         'bins': [0, 10],
#         'weights': [1, 10, 1000]
#     },
#     {
#         'bins': [0, 10],
#         'weights': [1, 100, 1000]
#     },
#     {
#         'bins': [0, 10],
#         'weights': [1, 100, 10000]
#     },
#     {
#         'bins': [0, 10],
#         'weights': [1, 1000, 10000]
#     },
#     {
#         'bins': [0, 1, 10],
#         'weights': [1, 10, 100, 1000]
#     },
#     {
#         'bins': [0, 1, 10],
#         'weights': [1, 10, 1000, 10000]
#     },
#     {
#         'bins': [0, 1, 10],
#         'weights': [1, 10, 1000, 100000]
#     },
#     {
#         'bins': [0, 1, 10],
#         'weights': [1, 100, 1000, 10000]
#     }
# ]

# for i, train_weights in enumerate(train_bins_weights):
#     experiments[f'mlp_weights_{i}'] = deepcopy(experiments['mlp'])
#     experiments[f'mlp_weights_{i}']['run_name'] = f'mlp_weights_{i}'
#     # experiments[f'mlp_weights_{i}']['data_module']['params'].update(
#     #     {
#     #         'label_bins' : train_weights['bins'],
#     #         'label_weights' : train_weights['weights']
#     #     }
#     # )
#     experiments[f'mlp_weights_{i}']['data_module']['params']['label_bins'] = train_weights['bins']
#     experiments[f'mlp_weights_{i}']['data_module']['params']['label_weights'] = train_weights['weights']
    
# for i, train_samples in enumerate(train_samples_cond):
#     experiments[f'mlp_samples_{i}'] = deepcopy(experiments['mlp'])
#     experiments[f'mlp_samples_{i}']['run_name'] = f'mlp_samples_{i}'
#     experiments[f'mlp_samples_{i}']['data_module']['params']['sample_bins'] = train_samples
    
