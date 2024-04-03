from models.vector.classification import MlpClassification, ClassificationModelModule, TransformerClassification
from models.vector.regression import MlpRegression, RegressionModelModule, TransformerRegression
from datasets.vector import classification as vector_classification
from datasets.vector import regression as vector_regression
import torch
from utils.callbacks import SaveImagePrediction, SaveVectorPrediction
from copy import deepcopy
from itertools import chain, combinations


def powerset(iterable, max = None):
    "powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if max is None:
        return chain.from_iterable(combinations(s, r) for r in range(0, len(s)+1))
    else:
        return chain.from_iterable(combinations(s, r) for r in range(0, max+1))

class default:
    #n_prev_times = 12
    time_0 = 24
    n_train_times = 48
    n_val_times = 48
    n_test_times = 48
    train_batch_size = 128
    train_num_workers = 8
    pred_batch_size = 2048
    train_sample_bins = None
    label_bins = None
    label_weights = None
    features_list = [
        'ArDS_4', 
        'Biweekly', 
        # 'AcAr', 
        'CtDS', 
        'DeAr', 
        # 'Cloud', 
        # 'OcDS', 
        'XQ', 
        'XArDS', 
        'XDeDS', 
        # 'DS', 
        #'DryMonths_0', 
        #'Coordinates_0,1', 
        # 'Distbd_0', #muito ruim
        # 'Distbd_1', #muito ruim
        # 'Distbd_2', #muito ruim
        # 'Distbd_3', #muito ruim
        # 'Distbd_4', #muito ruim
        # 'Dvd_0', #ruim
        # 'EF_0'
        # 'EF_1'
        # 'EF_2'
        # 'EF_3'
        # 'EF_4'
        # 'EF_5'
        # 'EF_6'
        ]  # first element is the target feature
    all_features_list = [
        'ArDS_12', 
        'Biweekly', 
        'AcAr', 
        'CtDS', 
        'DeAr', 
        'Cloud', 
        'OcDS', 
        'XQ', 
        'XArDS', 
        'XDeDS', 
        #'DS', 
        'DryMonths_0', 
        'Coordinates_0,1', 
        'Distbd_0', #muito ruim
        'Distbd_1', #muito ruim
        'Distbd_2', #muito ruim
        'Distbd_3', #muito ruim
        'Distbd_4', #muito ruim
        'Dvd_0', #ruim
        'EF_0',
        'EF_1',
        'EF_2',
        'EF_3',
        'EF_4',
        'EF_5',
        'EF_6',
        ] 
    

cls_vector_data_module = {
    'class': vector_classification.DataModule,
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
        'normalize_data' : True,
        'normalize_label' : False
    }
}

cls_vector_save_pred_callback = {
    'class' : SaveVectorPrediction,
    'params':{
        #'n_prev' : default.n_prev_times,
        'test_time_0': default.time_0, #  + default.n_train_times + default.n_val_times
        'test_times' : default.n_test_times + default.n_train_times + default.n_val_times
    }
}

reg_vector_data_module = {
    'class': vector_regression.ClassificationRegressionDataModule,
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
        'normalize_data' : True,
        'normalize_label' : False
    }
}

reg_vector_save_pred_callback = {
    'class' : SaveVectorPrediction,
    'params':{
        #'n_prev' : default.n_prev_times,
        'test_time_0': default.time_0 + default.n_train_times + default.n_val_times,
        'test_times' : default.n_test_times
    }
}

experiments = {
    'base':{
        'experiment_name': 'Deforestation Prediction',
        
        'cls_model_module': ClassificationModelModule,
        'reg_model_module': RegressionModelModule,
        
        'train_params':{
            'patience': 20,
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
    'cls_model': {
        'class': MlpClassification,
        'params':{
            'layers':[64, 256, 512, 256, 128]
        }
    },
    'cls_criterion': {
        'class' : torch.nn.CrossEntropyLoss,
        'params':{
            'reduction': 'none'
        }
    },        
    'cls_optimizer' : {
        'class' : torch.optim.Adam,
        'params':{
            'lr': 1e-6
        }
    },
    'reg_model': {
        'class': MlpRegression,
        'params':{
            'layers':[64, 256, 512, 256, 128]
        }
    },
    'reg_criterion': {
        'class' : torch.nn.MSELoss,
        'params':{
            'reduction': 'none'
        }
    },        
    'reg_optimizer' : {
        'class' : torch.optim.Adam,
        'params':{
            'lr': 2e-7
        }
    },
    'cls_data_module': cls_vector_data_module,
    'cls_save_pred_callback': cls_vector_save_pred_callback,
    'reg_data_module': reg_vector_data_module,
    'reg_save_pred_callback': reg_vector_save_pred_callback,
})

experiments['mlp'] = deepcopy(experiments['mlp_vector_base'])

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
    'cls_model': {
        'class': TransformerClassification,
        'params':{
            'n_layers': 6, 
            'd_model': 512, 
            'n_head': 8
        }
    },
    'cls_criterion': {
        'class' : torch.nn.CrossEntropyLoss,
        'params':{
            'reduction': 'none'
        }
    },        
    'cls_optimizer' : {
        'class' : torch.optim.Adam,
        'params':{
            'lr': 1e-6
        }
    },
    'reg_model': {
        'class': TransformerRegression,
        'params':{
            'n_layers': 6, 
            'd_model': 512, 
            'n_head': 8
        }
    },
    'reg_criterion': {
        'class' : torch.nn.MSELoss,
        'params':{
            'reduction': 'none'
        }
    },        
    'reg_optimizer' : {
        'class' : torch.optim.Adam,
        'params':{
            'lr': 2e-7
        }
    },
    'cls_data_module': cls_vector_data_module,
    'cls_save_pred_callback': cls_vector_save_pred_callback,
    'reg_data_module': reg_vector_data_module,
    'reg_save_pred_callback': reg_vector_save_pred_callback,
})

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