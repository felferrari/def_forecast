from pathlib import Path
from models.model_module import ModelModule
import segmentation_models_pytorch as smp
from lightning.pytorch.trainer.trainer import  Trainer
from lightning.pytorch.callbacks import EarlyStopping
from utils.ops import load_ml_image, load_sb_image, generate_images, generate_metric_figures, generate_histograms, evaluate_metric
import config
#import fire
import argparse
import mlflow
from mlflow.pytorch import autolog
import features 
from einops import rearrange
import matplotlib
import tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser()

parser.add_argument('function', type = str, choices=['train', 'predict', 'evaluate'])
parser.add_argument('model_name', type=str)
parser.add_argument('--version', type=int, required=False)

args = parser.parse_args()

default = config.default
experiments = config.experiments

def train(run_name):

    
    experiment = experiments[run_name]
    
    run_name = experiment['run_name']
    model_name = experiment['model_name']
    experiment_name = experiment['experiment_name']
    model = experiment['model']
    criterion = experiment['criterion']
    optimizer = experiment['optimizer']
    data_module_cfg = experiment['data_module']
    train_params = experiment['train_params']
    
    
    data_module = data_module_cfg['class'](**data_module_cfg['params'])
    model['params']['input_sample'] = data_module.train_dataloader().dataset[0]
    model = model['class'](**model['params'])
    criterion = criterion['class'](**criterion['params'])
    
    modelModule = ModelModule(model, criterion, optimizer['class'], optimizer['params'])
    
    callbacks = [
        EarlyStopping(
            monitor = 'val_loss',
            patience = train_params['patience'],
            verbose = True
        )
    ]
    trainer = Trainer(
        accelerator = train_params['accelerator'],
        
        logger = False,
        callbacks = callbacks,
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
        trainer.fit(
            model = modelModule,
            datamodule = data_module
        )
    
    
def predict(exp_name):    

    experiment = experiments[exp_name]
    
    run_name = experiment['run_name']
    experiment_name = experiment['experiment_name']
    data_module = experiment['data_module']
    save_pred_callback = experiment['save_pred_callback']
    pred_params = experiment['pred_params']
    
    save_pred_callback = save_pred_callback['class'](**save_pred_callback['params'])
    data_module = data_module['class'](**data_module['params'])
    
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
    data_module = experiment['data_module']
    
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
            
            predict_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path = dir, artifact_path= f'prediction/{run_name}.tif')
            #predict_path = list(Path(predict_path).glob('*.tif'))[0]
            predict_results = load_ml_image(predict_path)
            
            data_module.predict_dataloader()
            true_results = data_module.prediction_ds.dataset.label.data
            true_results = rearrange(data_module.prediction_ds.dataset.label.data, 'l (h w) -> h w l', h = mask.shape[0], w = mask.shape[1])
            true_results = true_results[:,:,-data_module.test_times:]
            
            generate_images(true_results, predict_results, mask)
            
            generate_metric_figures(true_results, predict_results, mask, mean_squared_error, f'MSE', exp_name, [1e-6, 6000])
            generate_metric_figures(true_results, predict_results, mask, mean_absolute_error, f'MAE', exp_name,  [1e-6, 600])
            
            generate_histograms(true_results, predict_results, mask, [0, 80], run_name)
            generate_histograms(true_results, predict_results, mask, [0, 1], run_name, normalize=True)
            
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