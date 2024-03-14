from models.model_module import ModelModule
import segmentation_models_pytorch as smp
from lightning.pytorch.trainer.trainer import  Trainer
from lightning.pytorch.callbacks import EarlyStopping
from utils.ops import evaluate_results, load_ml_image, load_sb_image
import config
import fire
import mlflow
from mlflow.pytorch import autolog
import features 
from einops import rearrange
import matplotlib

default = config.default
experiments = config.experiments

def train(exp_name):
    """
    The `train` function prepares and executes a machine learning experiment using the specified
    parameters and configurations.
    
    @param exp_name The `train` function you provided seems to be a script for training a machine
    learning model using PyTorch Lightning and MLflow. It takes an experiment name (`exp_name`) as input
    to specify which experiment configuration to use for training.
    """
    
    experiment = experiments[exp_name]
    
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
            
        mlflow.log_params(data_module_cfg['params'])
        #training
        trainer.fit(
            model = modelModule,
            datamodule = data_module
        )
    
    
def predict(exp_name, version = None):    
    """
    This Python function `predict` loads a trained PyTorch model from MLflow, performs testing and
    prediction using the model, and logs evaluation metrics using MLflow.
    
    @param exp_name The `exp_name` parameter in the `predict` function is used to specify the name of
    the experiment for which you want to make predictions. This experiment should be defined in the
    `experiments` dictionary that contains information about the experiment such as the run name,
    experiment name, data module, save
    @param version The `version` parameter in the `predict` function is used to specify a particular
    version of the experiment to run. If a version is provided, the function will search for runs within
    the specified experiment that match both the `run_name` and the provided `version`. If no version is
    provided (
    """
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
        mask = load_sb_image(features.mask_path)
        pred_results = save_pred_callback.final_image
        true_results = rearrange(data_module.prediction_ds.dataset.label.data, 'l (h w) -> h w l', h = mask.shape[0], w = mask.shape[1])
        true_results = true_results[:,:,data_module.n_previous_times:]
        
        
        mse, mae, norm_mse, norm_mae, mse__dict, mae__dict = evaluate_results(true_results, pred_results, mask, bins = [0, 1, 2, 5, 10, 80], run_name = exp_name)
        mlflow.log_metrics({
            'predicted_mse': mse,
            'predicted_mae': mae,
            'normalized_predicted_mse': norm_mse,
            'normalized_predicted_mae': norm_mae,
        })
        #mse__dict['Experiment Name'] = exp_name
        #mae__dict['Experiment Name'] = exp_name
        #mlflow.log_table(mse__dict, 'mse_hist.json')
        #mlflow.log_table(mae__dict, 'mae_hist.json')
    

if __name__ == '__main__':
    fire.Fire()