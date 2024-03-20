from models.models import Resunet, Mlp, TransformerVector
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
    train_batch_size = 128
    train_num_workers = 4
    pred_batch_size = 2048
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

experiments['mlp_vector_base'] = deepcopy(experiments['base'])
experiments['mlp_vector_base'].update({
    'model': {
        'class': Mlp,
        'params':{
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

experiments['mlp'] = deepcopy(experiments['mlp_vector_base'])

experiments['zeros'] = deepcopy(experiments['mlp'])

#base Experiment
experiments['mlp_0'] = deepcopy(experiments['mlp'])

#Reweighting
experiments['mlp_1'] = deepcopy(experiments['mlp'])
experiments['mlp_1']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]

#Resampling
experiments['mlp_2'] = deepcopy(experiments['mlp'])
experiments['mlp_2']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]

#Reweighting and Resampling
experiments['mlp_3'] = deepcopy(experiments['mlp'])
experiments['mlp_3']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]
experiments['mlp_3']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]

#reweighting modificado
experiments['test'] = deepcopy(experiments['mlp'])
#experiments['test']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]
#experiments['test']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]
experiments['test']['data_module']['params']['normalize_data'] = False

# #Transformer
# experiments['transformer_vector_base'] = deepcopy(experiments['base'])
# experiments['transformer_vector_base'].update({
#     'run_name': 'transformer',
#     'model_name': 'transformer',
#     'model': {
#         'class': TransformerVector,
#         'params':{
#             'n_layers': 6,
#             'n_head': 8,
#             'd_model': 512
#         }            
#     },
#     'optimizer' : {
#         'class' : torch.optim.Adam,
#         'params':{
#             'lr': 2e-7
#         }
#     },
#     'data_module': vector_data_module,
#     'save_pred_callback': vector_save_pred_callback,
# })
# experiments['transformer_vector_base']['data_module']['params'].update({'train_batch_size' : 128})
# experiments['transformer_vector_base']['data_module']['params'].update({'pred_batch_size' : 2048})
# experiments['transformer_vector_base']['data_module']['params'].update({'train_num_workers' : 4})

# experiments['transformer'] = deepcopy(experiments['transformer_vector_base'])
# experiments['transformer']['run_name'] = 'transformer'
# # experiments['transformer']['criterion'] = {
# #     'class': RegressionFocalLoss,
# #     'params':{
# #         'alpha': 2,
# #         'beta': 1,
# #         'reduction': 'none'
# #     }
# # }

# #experiments['transformer']['data_module']['params']['sample_bins'] = [0, 1]
# #experiments['transformer']['data_module']['params']['label_bins'] = [0]

# experiments['transformer_0'] = deepcopy(experiments['transformer_vector_base'])
# experiments['transformer_0']['run_name'] = 'transformer_0'


# experiments['transformer_1'] = deepcopy(experiments['transformer_vector_base'])
# experiments['transformer_1']['run_name'] = 'transformer_1'
# experiments['transformer_1']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]

# experiments['transformer_2'] = deepcopy(experiments['transformer_vector_base'])
# experiments['transformer_2']['run_name'] = 'transformer_2'
# experiments['transformer_2']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]

# experiments['transformer_3'] = deepcopy(experiments['transformer_vector_base'])
# experiments['transformer_3']['run_name'] = 'transformer_3'
# experiments['transformer_3']['data_module']['params']['label_bins'] = [0, 1, 2, 5, 10]
# experiments['transformer_3']['data_module']['params']['sample_bins'] = [0, 1, 2, 5, 10]

# experiments['transformer_4'] = deepcopy(experiments['transformer_vector_base'])
# experiments['transformer_4']['run_name'] = 'transformer_4'
# experiments['transformer_4']['data_module']['params']['sample_bins'] = [0]

# experiments['transformer_5'] = deepcopy(experiments['transformer_vector_base'])
# experiments['transformer_5']['run_name'] = 'transformer_5'
# experiments['transformer_5']['data_module']['params']['sample_bins'] = [0]
# experiments['transformer_5']['data_module']['params']['features_list'] = ['ArDS_12', 'Biweekly', 'DeAr', 'XQ', 'XArDS', 'XDeDS'] 