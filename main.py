from models.model_module import ModelModule
import segmentation_models_pytorch as smp
import torch
from utils.image_datasets import ImageDataModule
from utils.callbacks import SavePrediction
from lightning.pytorch.trainer.trainer import  Trainer
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from models.models import Resunet
from utils.ops import evaluate_results, load_ml_image, load_sb_image
import config, paths
from pathlib import Path
import fire
import mlflow
from mlflow.pytorch import autolog

default = config.default
experiments = config.experiments
def train(
    exp_name,
    ):
    
    #exp_path = Path(f'experiments/exp_{exp_name}')
    #exp_path.mkdir(exist_ok=True)
    
    
    #Data setup
    # data_module = ImageDataModule(
    #     n_previous_times = n_prev_times,
    #     train_times = n_train_times,
    #     val_times = n_val_times,
    #     test_times = n_test_times,
    #     patch_size = patch_size,
    #     train_overlap = train_overlap,
    #     train_batch_size = train_batch_size,
    #     train_num_workers = train_num_workers,
    #     prediction_overlap = prediction_overlap,
    #     features = features
    # )
    
    experiment = experiments[exp_name]
    
    run_name = experiment['run_name']
    model_name = experiment['model_name']
    experiment_name = experiment['experiment_name']
    model = experiment['model']
    criterion = experiment['criterion']
    optimizer = experiment['optimizer']
    optimizer_params = experiment['optimizer_params']
    data_module = experiment['data_module']
    
    #Model setup
    # model = smp.UnetPlusPlus(
    #     #encoder_name="resnext101_32x8d",
    #     encoder_name="resnet34",
    #     in_channels=n_prev,
    #     classes = 1,
    #     encoder_weights = None,
    #     activation='identity'
    # )

    
    modelModule = ModelModule(model, criterion, optimizer, optimizer_params)
    
    callbacks = [
        EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            verbose = True
        )
    ]
    trainer = Trainer(
        accelerator = 'gpu',
        logger = False,
        callbacks = callbacks,
        #max_epochs=2
    )
    
    
    mlflow.set_experiment(experiment_name)
    autolog(
        registered_model_name = model_name
        )
    with mlflow.start_run(run_name=run_name, log_system_metrics = True):
        runs = mlflow.search_runs(
            experiment_names=[experiment_name], 
            filter_string = f'run_name = "{run_name}"', 
            order_by = ['params.run_version DESC']
            )
        if len(runs) == 1:
            mlflow.log_param('run_version', 1)
        else:
            mlflow.log_param('run_version', int(runs['params.run_version'][0])+1)
            
        
        #training
        trainer.fit(
            model = modelModule,
            datamodule = data_module
        )
    
    
def predict(
    exp_name,
    version = None
):    
    experiment = experiments[exp_name]
    
    run_name = experiment['run_name']
    model_name = experiment['model_name']
    experiment_name = experiment['experiment_name']
    data_module = experiment['data_module']
    save_pred_callback = experiment['save_pred_callback']
    
    callbacks = [
        save_pred_callback
    ]
    trainer = Trainer(
        accelerator = 'gpu',
        logger = False,
        callbacks = callbacks,
        #max_epochs=2
    )
    
    mlflow.set_experiment('Deforestation Prediction')
    
    if version is None:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string = f'run_name = "{run_name}"',
            order_by=['params.version DESC']
        )
    else:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string = f'run_name = "{run_name}" AND params.version = "{version}"',
            order_by=['params.version DESC']
        )
    run_id = runs['run_id'][0]
    model_id = f'runs:/{run_id}/model'
    modelModule = mlflow.pytorch.load_model(model_id)
    
    #model_file_path = Path(f'experiments/exp/model.ckpt')
    #modelModule = ModelModule.load_from_checkpoint(model_file_path)
    mlflow.set_experiment(experiment_name)
    autolog()
    with mlflow.start_run(run_id=run_id):
        #Test
        test_results = trainer.test(
            model = modelModule,
            datamodule = data_module
        )
        mlflow.log_metrics(test_results[0])
        
        #Prediction
        trainer.predict(
            model = modelModule,
            datamodule = data_module,
            return_predictions = False
        )
    
        #results
        pred_results = save_pred_callback.final_image
        qd_paths_dict = paths.path_to_data['def_data']
        ref_data = load_ml_image(qd_paths_dict)[:,:,-22:]
        #path_to_save = f'experiments/exp_0/images'
        mask = load_sb_image(paths.path_to_mask)
        evaluate_results(ref_data, pred_results, mask, bins = [0, 1, 2, 5, 10, 30, 100])
    
  

if __name__ == '__main__':
    fire.Fire()