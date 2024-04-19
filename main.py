from models.model_module import ModelModule
import argparse
from lightning.pytorch.trainer.trainer import Trainer
from utils.ops import load_yaml
from factory import build
import mlflow
from mlflow.pytorch import autolog
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import tempfile
from callbacks.callbacks import SaveVectorPrediction
from utils.ops import load_ml_image, load_sb_image, generate_images, generate_metric_figures, generate_histograms, evaluate_metric, flatten_dict, integrated_gradients
from sklearn.metrics import mean_squared_error, mean_absolute_error
import features 
from einops import rearrange
from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays
import geopandas as gpd


parser = argparse.ArgumentParser()

parser.add_argument('function', type = str, choices=['train', 'predict', 'evaluate'])
parser.add_argument('model_name', type=str)

args = parser.parse_args()

experiments = load_yaml('experiments.yaml')

#default = config.default
#experiments = config.experiments

def train(run_name):
    
    runs = mlflow.search_runs(
        filter_string = f'run_name = "{run_name}"'
        )
    
    for run_id in runs['run_id']:
        mlflow.delete_run(run_id=run_id)
    
    autolog(log_models=False)
    
    with mlflow.start_run(run_name=run_name, log_system_metrics = True):
        
        with tempfile.TemporaryDirectory() as temp_dir:
    
            run_description, model, data_module, criterion, optimizer, optimizer_params, train_params, _ = build(experiments, run_name)
            
            mlflow.log_params(flatten_dict(run_description))
            
            model_module = ModelModule(
                model = model,
                criterion = criterion,
                optimizer = optimizer,
                optimizer_params = optimizer_params
            )
            
            early_stop = EarlyStopping(
                monitor = 'val_loss',
                patience = train_params['early_stop_patience'],
                verbose = True
            )
            
            model_checkpoint = ModelCheckpoint(
                dirpath = temp_dir,
                filename = f'model_{run_name}',
                monitor = 'val_loss',
                verbose = True
            )
            
            callbacks = [
                early_stop,
                model_checkpoint
            ]
            
            
            trainer = Trainer(
                accelerator='gpu',
                logger = False,
                enable_progress_bar = False,
                callbacks = callbacks,
                max_epochs = train_params['max_epochs']
                #limit_train_batches=10,
                #limit_val_batches=10,
            )

            trainer.fit(model = model_module, datamodule=data_module)
            
            model = ModelModule.load_from_checkpoint(model_checkpoint.best_model_path)
            
            mlflow.pytorch.log_model(model, 'model')

def predict(run_name):
    _, _, data_module, _, _, _, _, predict_params = build(experiments, run_name)
    
    save_pred_callback = SaveVectorPrediction(
        lag_0=predict_params['lag_0'],
        lag_size=predict_params['lag_size'],
        log_tiff = True
    )
    
    callbacks = [
        save_pred_callback
    ]
    trainer = Trainer(
        accelerator = 'gpu',
        logger = False,
        enable_progress_bar=False,
        callbacks = callbacks,
    )
    
    runs = mlflow.search_runs(
            filter_string = f'run_name = "{run_name}"',
        )
    run_id = runs['run_id'][0]
    
    model_id = f'runs:/{run_id}/model'
    modelModule = mlflow.pytorch.load_model(model_id)
    
    autolog(log_models=False)
    
    with mlflow.start_run(run_id=run_id):
                trainer.predict(
                    model = modelModule,
                    datamodule = data_module,
                    return_predictions = False
                )
                
    
def evaluate(run_name):    

    _, _, data_module, _, _, _, _, predict_params = build(experiments, run_name)
    
    runs = mlflow.search_runs(
            filter_string = f'run_name = "{run_name}"',
        )
    run_id = runs['run_id'][0]
    
    model_id = f'runs:/{run_id}/model'
    model = mlflow.pytorch.load_model(model_id).model
    

    
    autolog()
    with mlflow.start_run(run_id=run_id):
        with tempfile.TemporaryDirectory() as dir:
            
            mask = load_sb_image(features.mask_path)
            
            predict_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path = dir, artifact_path= f'prediction/{run_name}.tif')
            #predict_path = list(Path(predict_path).glob('*.tif'))[0]
            predict_results = load_ml_image(predict_path)
            
            data_module.predict_dataloader()
            true_results = data_module.predict_ds.feature_dataset.label.data
            true_results = rearrange(data_module.predict_ds.feature_dataset.label.data, 'l (h w) -> h w l', h = mask.shape[0], w = mask.shape[1])
            true_results = true_results[:,:,-data_module.test_times:]
            
            generate_images(true_results, predict_results, mask, percentile=95)
            generate_images(true_results, predict_results, mask, percentile=99)
            generate_images(true_results, predict_results, mask, percentile=99.9)
            generate_images(true_results, predict_results, mask, percentile=99.99)
            generate_images(true_results, predict_results, mask)
            
            generate_metric_figures(true_results, predict_results, mask, mean_squared_error, f'MSE', run_name, [1e-6, 6000])
            generate_metric_figures(true_results, predict_results, mask, mean_absolute_error, f'MAE', run_name,  [1e-6, 600])
            
            generate_histograms(true_results, predict_results, mask, [0, 30], run_name)
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
            
            norm_mse = evaluate_metric(true_results, predict_results, mask, mean_squared_error, True)
            norm_mae = evaluate_metric(true_results, predict_results, mask, mean_absolute_error, True)
            
            integrated_gradients(
                model=model,
                dataloader=data_module.test_dataloader(),
                device = 'cuda:0',
                run_name = run_name)
            
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
                
                'norm_mse': norm_mse,
                'norm_mae': norm_mae,
                
            })

            
if __name__ == '__main__':
    locals()[args.function](args.model_name)