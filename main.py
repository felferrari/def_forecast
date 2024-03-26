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
        model_checkpoint = ModelCheckpoint(
            dirpath = temp_dir,
            filename= 'classification_model',
            verbose = True,
            monitor= 'val_loss',
            mode = 'min'
            )
        
        callbacks = [
            early_stop,
            model_checkpoint
        ]
        
        mlflow.set_experiment(experiment_name)
        
        runs = mlflow.search_runs(
            experiment_names=[experiment_name], 
            filter_string = f'run_name = "{run_name}"'
            )
        
        for run_id in runs['run_id']:
            mlflow.delete_run(run_id=run_id)

        with mlflow.start_run(run_name=run_name):
                
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
    data_module = data_module_cfg['class'](**data_module_cfg['params'])
    
    callbacks = [
        save_pred_callback
    ]
    trainer = Trainer(
        accelerator = pred_params['accelerator'],
        logger = False,
        callbacks = callbacks,
    )
    
    mlflow.set_experiment(experiment_name)
    
    runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string = f'run_name = "{run_name}"',
        )
    run_id = runs['run_id'][0]
    model_id = f'runs:/{run_id}/classification_model'
    model_module = mlflow.pytorch.load_model(model_id)
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_id=run_id):
        autolog()
        trainer.predict(
            model = model_module,
            datamodule = data_module,
            return_predictions = False
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            #tmp_file = Path(tmp_dir) / f'{mlflow.active_run().info.run_name}_{uuid.uuid4()}.tif'
            tmp_file = Path(tmp_dir) / f'{mlflow.active_run().info.run_name}_full.tif'
            save_geotiff(features.mask_path, tmp_file, save_pred_callback.final_image, 'float')
            mlflow.log_artifact(tmp_file, 'predictions')
    
    
def evaluate_cls(run_name):    

    experiment = experiments[run_name]
    
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
            
            predict_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path = dir, artifact_path= f'predictions/{run_name}_full.tif')
            #predict_path = list(Path(predict_path).glob('*.tif'))[0]
            predict_results = load_ml_image(predict_path)
            
            data_module.predict_dataloader()
            true_results = data_module.prediction_ds.dataset.label.data
            true_results = rearrange(data_module.prediction_ds.dataset.label.data, 'l (h w) -> h w l', h = mask.shape[0], w = mask.shape[1])
            true_results = true_results[:,:,-data_module.test_times:]
            true_results = (true_results > 0).astype(np.int8)
            predict_results = predict_results[:,:,-data_module.test_times:]
            
            true_results = np.expand_dims(true_results.max(axis=-1), axis=-1)
            predict_results = np.expand_dims(predict_results.max(axis=-1), axis=-1)
            
            generate_images(true_results, predict_results, mask)
            
            
        mlflow.log_metrics({
            'pred_f1-score': f1_score(true_results.flatten(), predict_results.flatten()),
            'pred_precision': precision_score(true_results.flatten(), predict_results.flatten()),
            'pred_recall': recall_score(true_results.flatten(), predict_results.flatten())
        })
        
      

def train_reg(run_name):    
    
    experiment = experiments[run_name]
    experiment_name = experiment['experiment_name']
    model_module = experiment['reg_model_module']
    
    model = experiment['reg_model']
    criterion = experiment['reg_criterion']
    optimizer = experiment['reg_optimizer']
    
    data_module_cfg = experiment['reg_data_module']
    
    train_params = experiment['train_params']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        early_stop = EarlyStopping(
            monitor = 'val_loss',
            patience = train_params['patience'],
            verbose = True
        )
        model_checkpoint = ModelCheckpoint(
            dirpath = temp_dir,
            filename= 'classification_model',
            verbose = True,
            monitor= 'val_loss',
            mode = 'min'
            )
        
        callbacks = [
            early_stop,
            model_checkpoint
        ]
    
        mlflow.set_experiment(experiment_name)
        
        runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string = f'run_name = "{run_name}"',
            )
        run_id = runs['run_id'][0]
        
        mlflow.set_experiment(experiment_name)
        
        predict_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path = temp_dir, artifact_path= f'predictions/{run_name}_full.tif')
        predict_results = load_ml_image(predict_path)
        data_module_cfg['params']['cls_mask'] = predict_results
        
        data_module = data_module_cfg['class'](**data_module_cfg['params'])
        model['params']['input_sample'] = data_module.train_dataloader().dataset[0]
        
        model = model['class'](**model['params'])
        
        criterion = criterion['class'](**criterion['params'])
        
        modelModule = model_module(model, criterion, optimizer['class'], optimizer['params'])
        
        with mlflow.start_run(run_id=run_id):
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
                'regression_model'
                )
            mlflow.log_metric('best_reg_val_loss', model_checkpoint.best_model_score)

def predict_reg(run_name):    

    experiment = experiments[run_name]
    experiment_name = experiment['experiment_name']
    
    data_module_cfg = experiment['reg_data_module']
    
    pred_params = experiment['pred_params']
    
    save_pred_callback = experiment['reg_save_pred_callback']
    
    save_pred_callback = save_pred_callback['class'](**save_pred_callback['params'])
    

    
    callbacks = [
        save_pred_callback
    ]
    trainer = Trainer(
        accelerator = pred_params['accelerator'],
        logger = False,
        callbacks = callbacks,
    )
    
    mlflow.set_experiment(experiment_name)
    
    runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string = f'run_name = "{run_name}"',
        )
    run_id = runs['run_id'][0]
    model_id = f'runs:/{run_id}/regression_model'
    model_module = mlflow.pytorch.load_model(model_id)
    
    mlflow.set_experiment(experiment_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
    
        predict_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path = temp_dir, artifact_path= f'predictions/{run_name}_full.tif')
        predict_results = load_ml_image(predict_path)
        data_module_cfg['params']['cls_mask'] = predict_results
        data_module = data_module_cfg['class'](**data_module_cfg['params'])
        
        with mlflow.start_run(run_id=run_id):
            autolog()
            
            trainer.predict(
                model = model_module,
                datamodule = data_module,
                return_predictions = False
            )
            
            #tmp_file = Path(tmp_dir) / f'{mlflow.active_run().info.run_name}_{uuid.uuid4()}.tif'
            tmp_file = Path(temp_dir) / f'{mlflow.active_run().info.run_name}_final.tif'
            save_geotiff(features.mask_path, tmp_file, save_pred_callback.final_image, 'float')
            mlflow.log_artifact(tmp_file, 'predictions')
        

def evaluate_reg(run_name):    

    experiment = experiments[run_name]
    
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
            
            mse = evaluate_metric(true_results, predict_results, mask, mean_squared_error, False)
            norm_mse = evaluate_metric(true_results, predict_results, mask, mean_squared_error, True)
            
            mae = evaluate_metric(true_results, predict_results, mask, mean_absolute_error, False)
            norm_mae = evaluate_metric(true_results, predict_results, mask, mean_absolute_error, True)
            
            mlflow.log_metrics({
                'predicted_mse': mse,
                'predicted_mae': mae,
                'normalized_predicted_mse': norm_mse,
                'normalized_predicted_mae': norm_mae,
            })
  

if __name__ == '__main__':
    locals()[args.function](args.model_name)