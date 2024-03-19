from models.models import Resunet, Mlp, TransformerVector
from models.losses import RegressionFocalLoss
from utils.image_datasets import ImageDataModule
from utils.vector_datasets import VectorDataModule
import torch
from utils.callbacks import SaveImagePrediction, SaveVectorPrediction
import models
import utils
from copy import deepcopy

class default:
    #n_prev_times = 12
    time_0 = 24
    n_train_times = 48
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
    features_list = [
        'ArDS_12', 
        #'Biweekly', 
        #'AcAr', 
        #'CtDS', 
        #'DeAr', 
        #'Cloud', 
        #'OcDS', 
        #'XQ', 
        #'XArDS', 
        #'XDeDS', 
        #'DS', 
        #'DryMonths_0', 
        #'Coordinates_0,1', 
        #'Distbd_0', #muito ruim
        #'Distbd_1', #muito ruim
        #'Distbd_2', #muito ruim
        #'Distbd_3', #muito ruim
        #'Distbd_4', #muito ruim
        #'Dvd_0', #ruim
        #'EF_0'
        #'EF_1'
        #'EF_2'
        #'EF_3'
        #'EF_4'
        #'EF_5'
        #'EF_6'
        ]  # first element is the target feature

image_data_module = {
    'class': ImageDataModule,
    'params':{
        #'n_previous_times' : default.n_prev_times,
        'time_0' : default.time_0,
        'train_times' : default.n_train_times,
        'val_times' : default.n_val_times,
        'test_times' : default.n_test_times,
        'test_time_0': default.time_0 + default.n_train_times + default.n_val_times,
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
        #'n_previous_times' : default.n_prev_times,
        'time_0' : default.time_0,
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
        #'n_prev' : default.n_prev_times,
        'patch_size' : default.patch_size,
        'test_times' : default.n_test_times,
        'test_time_0': default.time_0 + default.n_train_times + default.n_val_times,
        'border_removal' : default.prediction_border_removal,
        'log_tiff' : True
    }
}

vector_save_pred_callback = {
    'class' : SaveVectorPrediction,
    'params':{
        #'n_prev' : default.n_prev_times,
        'test_times' : default.n_test_times,
        'test_time_0': default.time_0 + default.n_train_times + default.n_val_times,
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
            'limit_val_batches': 1000,
            
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
            #'in_depth' : default.n_prev_times,
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

#reweighting modificado
experiments['test'] = deepcopy(experiments['mlp'])
experiments['test']['run_name'] = 'test'
#experiments['test']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]
#experiments['test']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]
experiments['test']['data_module']['params']['normalize_data'] = False

                                    
#Transformer
experiments['transformer_vector_base'] = deepcopy(experiments['base'])
experiments['transformer_vector_base'].update({
    'run_name': 'transformer',
    'model_name': 'transformer',
    'model': {
        'class': TransformerVector,
        'params':{
            'n_layers': 6,
            'n_head': 8,
            'd_model': 512
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
experiments['transformer_vector_base']['data_module']['params'].update({'train_batch_size' : 128})
experiments['transformer_vector_base']['data_module']['params'].update({'pred_batch_size' : 2048})
experiments['transformer_vector_base']['data_module']['params'].update({'train_num_workers' : 4})

experiments['transformer'] = deepcopy(experiments['transformer_vector_base'])
experiments['transformer']['run_name'] = 'transformer'
# experiments['transformer']['criterion'] = {
#     'class': RegressionFocalLoss,
#     'params':{
#         'alpha': 2,
#         'beta': 1,
#         'reduction': 'none'
#     }
# }

#experiments['transformer']['data_module']['params']['sample_bins'] = [0, 1]
#experiments['transformer']['data_module']['params']['label_bins'] = [0]

experiments['transformer_0'] = deepcopy(experiments['transformer_vector_base'])
experiments['transformer_0']['run_name'] = 'transformer_0'


experiments['transformer_1'] = deepcopy(experiments['transformer_vector_base'])
experiments['transformer_1']['run_name'] = 'transformer_1'
experiments['transformer_1']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]

experiments['transformer_2'] = deepcopy(experiments['transformer_vector_base'])
experiments['transformer_2']['run_name'] = 'transformer_2'
experiments['transformer_2']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]

experiments['transformer_3'] = deepcopy(experiments['transformer_vector_base'])
experiments['transformer_3']['run_name'] = 'transformer_3'
experiments['transformer_3']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]
experiments['transformer_3']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]

experiments['transformer_4'] = deepcopy(experiments['transformer_vector_base'])
experiments['transformer_4']['run_name'] = 'transformer_4'
experiments['transformer_4']['data_module']['params']['sample_bins'] = [0]

experiments['transformer_5'] = deepcopy(experiments['transformer_vector_base'])
experiments['transformer_5']['run_name'] = 'transformer_5'
experiments['transformer_5']['data_module']['params']['sample_bins'] = [0]
experiments['transformer_5']['data_module']['params']['features_list'] = ['ArDS_12', 'Biweekly', 'DeAr', 'XQ', 'XArDS', 'XDeDS'] 