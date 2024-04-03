from models.vector.classification import MlpClassification, ClassificationModelModule, TransformerClassification
from models.vector.regression import MlpRegression, RegressionModelModule, TransformerRegression
from datasets.vector import classification as vector_classification
from datasets.vector import regression as vector_regression
import torch
from utils.callbacks import SaveImagePrediction, SaveVectorPrediction
from copy import deepcopy

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
            'patience': 10,
            'accelerator' : 'gpu',
            'limit_train_batches': None,
            'limit_val_batches': None,
            
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
            'lr': 5e-7
        }
    },
    'cls_data_module': cls_vector_data_module,
    'cls_save_pred_callback': cls_vector_save_pred_callback,
    'reg_data_module': reg_vector_data_module,
    'reg_save_pred_callback': reg_vector_save_pred_callback,
})

experiments['transformer'] = deepcopy(experiments['transformer_vector_base'])

#base Experiment
experiments['transformer_0'] = deepcopy(experiments['transformer'])
