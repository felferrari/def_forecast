from models.models import Resunet
from utils.image_datasets import ImageDataModule
import torch
from utils.callbacks import SavePrediction



class default:
    n_prev_times = 2
    n_train_times = 34
    n_val_times = 24
    n_test_times = 24
    patch_size = 32
    train_overlap = 0.8
    train_batch_size = 32
    train_num_workers = 1
    prediction_overlap = 0.5
    prediction_border_removal = 4
    features = ['ArCS', 'HIDR', 'Monthly']

image_data_module = ImageDataModule(
    n_previous_times = default.n_prev_times,
    train_times = default.n_train_times,
    val_times = default.n_val_times,
    test_times = default.n_test_times,
    patch_size = default.patch_size,
    train_overlap = default.train_overlap,
    train_batch_size = default.train_batch_size,
    train_num_workers = default.train_num_workers,
    prediction_overlap = default.prediction_overlap,
    features = default.features
)

save_pred_callback = SavePrediction(
    n_prev = default.n_prev_times,
    patch_size = default.patch_size,
    test_times = default.n_test_times,
    border_removal=default.prediction_border_removal,
    save_tiff=True
)
    
experiments = {
    'resunet':{
        'experiment_name': 'Deforestation Prediction',
        'run_name': 'resunet',
        'model_name': 'resunet',
        'model': Resunet(
            in_depth= default.n_prev_times,
            depths = [32, 64, 128, 256],
            ),
        'criterion': torch.nn.MSELoss(reduction='none'),
        'optimizer' : torch.optim.Adam,
        'optimizer_params': {
            'lr': 1e-5
        },
        'data_module': image_data_module,
        'save_pred_callback': save_pred_callback
        
    }
}