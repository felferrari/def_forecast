#from models.model_module import ModelModule
from lightning.pytorch.trainer.trainer import  Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils.ops import load_ml_image, load_sb_image, generate_images, generate_metric_figures, generate_histograms, evaluate_metric, save_geotiff
import config
import argparse
import mlflow
from mlflow.pytorch import autolog, log_model
import datasets.features as features 
from einops import rearrange
import tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score
import tempfile
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('function', type = str, choices=['train_cls', 'predict_cls', 'evaluate_cls', 'train_reg', 'predict_reg', 'evaluate_reg'])
parser.add_argument('model_name', type=str)

args = parser.parse_args()

default = config.default
experiments = config.experiments

def train_cls(run_name):
    experiment = experiments[run_name]
    experiment_name = experiment['experiment_name']
    model_module = experiment['cls_model_module']
    
    model = experiment['cls_model']
    criterion = experiment['cls_criterion']
    optimizer = experiment['cls_optimizer']
    
    data_module_cfg = experiment['cls_data_module']
    
    train_params = experiment['train_params']
    
    data_module = data_module_cfg['class'](**data_module_cfg['params'])
    model['params']['input_sample'] = data_module.train_dataloader().dataset[0]
    
    model = model['class'](**model['params'])
    
    criterion = criterion['class'](**criterion['params'])
    
    modelModule = model_module(model, criterion, optimizer['class'], optimizer['params'])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        early_stop = EarlyStopping(
            monitor = 'val_loss',
            patience = train_params['patience'],
            verbose = True
        )
    ]
    trainer = Trainer(
        accelerator = train_params['accelerator'],
        
        logger = False,
        callbacks = callbacks,
        enable_progress_bar = False,
        limit_train_batches=train_params['limit_train_batches'],
        limit_val_batches=train_params['limit_val_batches']
        #max_epochs=2
    )
    
    
    mlflow.set_experiment(experiment_name)
    
    runs = mlflow.search_runs(
        experiment_names=[experiment_name], 
        filter_string = f'run_name = "{run_name}"'
        )
    
    for run_id in runs['run_id']:
        mlflow.delete_run(run_id=run_id)
    
    autolog(
        registered_model_name = f'model_{run_name}'
        )
    with mlflow.start_run(run_name=run_name, log_system_metrics = True):
            
        mlflow.log_params(data_module_cfg['params'])
        #training
            autolog(log_models=False, checkpoint=False) #registered_model_name = f'model_{run_name}_cls')
        #mlf_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name)
            
            trainer = Trainer(
                accelerator = train_params['accelerator'],
                logger = False,
                callbacks = callbacks,
                limit_train_batches=train_params['limit_train_batches'],
                limit_val_batches=train_params['limit_val_batches'],
                #max_epochs=2
            )
            trainer.fit(
                model = modelModule,
                datamodule = data_module
            )
            
            log_model(
                model_module.load_from_checkpoint(model_checkpoint.best_model_path),
                'classification_model'
                )
            mlflow.log_metric('best_cls_val_loss', model_checkpoint.best_model_score)
            
            data_module
            
    
def predict_cls(run_name):    

    experiment = experiments[run_name]
    experiment_name = experiment['experiment_name']
    
    data_module_cfg = experiment['cls_data_module']
    
    pred_params = experiment['pred_params']
    
    save_pred_callback = experiment['cls_save_pred_callback']
    
    save_pred_callback = save_pred_callback['class'](**save_pred_callback['params'])
    data_module = data_module['class'](**data_module['params'])
    
    callbacks = [
        save_pred_callback
    ]
    trainer = Trainer(
        accelerator = pred_params['accelerator'],
        logger = False,
        enable_progress_bar=False,
        callbacks = callbacks,
    )
    
    mlflow.set_experiment(experiment_name)
    
    runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string = f'run_name = "{run_name}"',
        )
    run_id = runs['run_id'][0]
    model_id = f'runs:/{run_id}/model'
    modelModule = mlflow.pytorch.load_model(model_id)
    
    mlflow.set_experiment(experiment_name)
    autolog()
    with mlflow.start_run(run_id=run_id):
        #Test
        # test_results = trainer.test(
        #     model = modelModule,
        #     datamodule = data_module
        # )
        # mlflow.log_metrics(test_results[0])
        
        #Prediction
        trainer.predict(
            model = modelModule,
            datamodule = data_module,
            return_predictions = False
        )
    
def evaluate(exp_name):    

    experiment = experiments[exp_name]
    
    run_name = experiment['run_name']
    experiment_name = experiment['experiment_name']
    data_module = experiment['cls_data_module']
    
    data_module = data_module['class'](**data_module['params'])
    
    mlflow.set_experiment(experiment_name)
    
    runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string = f'run_name = "{run_name}"',
        )
    run_id = runs['run_id'][0]
    
    mlflow.set_experiment(experiment_name)
    autolog()
    with mlflow.start_run(run_id=run_id):
        with tempfile.TemporaryDirectory() as dir:
            
            mask = load_sb_image(features.mask_path)
            
            predict_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path = dir, artifact_path= f'predictions/{run_name}_final.tif')
            #predict_path = list(Path(predict_path).glob('*.tif'))[0]
            predict_results = load_ml_image(predict_path)
            
            data_module.predict_dataloader()
            true_results = data_module.prediction_ds.dataset.label.data
            true_results = rearrange(data_module.prediction_ds.dataset.label.data, 'l (h w) -> h w l', h = mask.shape[0], w = mask.shape[1])
            true_results = true_results[:,:,-data_module.test_times:]
            predict_results = predict_results[:,:,-data_module.test_times:]
            
            
            generate_images(true_results, predict_results, mask)
            
            generate_metric_figures(true_results, predict_results, mask, mean_squared_error, f'MSE', run_name, [1e-6, 6000])
            generate_metric_figures(true_results, predict_results, mask, mean_absolute_error, f'MAE', run_name,  [1e-6, 600])
            
            generate_histograms(true_results, predict_results, mask, [0, 80], run_name)
            generate_histograms(true_results, predict_results, mask, [0, 1], run_name, normalize=True)
            
            mse = evaluate_metric(true_results, predict_results, mask, mean_squared_error, False)
            
            
            mae = evaluate_metric(true_results, predict_results, mask, mean_absolute_error, False)
            
            
            norm_mse_95 = evaluate_metric(true_results, predict_results, mask, mean_squared_error, True, 95)
            norm_mae_95 = evaluate_metric(true_results, predict_results, mask, mean_absolute_error, True, 95)
            
            norm_mse_99 = evaluate_metric(true_results, predict_results, mask, mean_squared_error, True, 99)
            norm_mae_99 = evaluate_metric(true_results, predict_results, mask, mean_absolute_error, True, 99)
            
            norm_mse_999 = evaluate_metric(true_results, predict_results, mask, mean_squared_error, True, 99.9)
            norm_mae_999 = evaluate_metric(true_results, predict_results, mask, mean_absolute_error, True, 99.9)
            
            norm_mse_9999= evaluate_metric(true_results, predict_results, mask, mean_squared_error, True, 99.99)
            norm_mae_9999= evaluate_metric(true_results, predict_results, mask, mean_absolute_error, True, 99.99)
            
            mlflow.log_metrics({
                'predicted_mse': mse,
                'predicted_mae': mae,
                'norm_mse_95': norm_mse_95,
                'norm_mae_95': norm_mae_95,
                
                'norm_mse_99': norm_mse_99,
                'norm_mae_99': norm_mae_99,
                
                'norm_mse_999': norm_mse_999,
                'norm_mae_999': norm_mae_999,
                
                'norm_mse_9999': norm_mse_9999,
                'norm_mae_9999': norm_mae_9999,
                
            })
  

if __name__ == '__main__':
    locals()[args.function](args.model_name)