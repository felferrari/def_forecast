from models.model_module import ModelModule
import segmentation_models_pytorch as smp
from lightning.pytorch.trainer.trainer import  Trainer
from lightning.pytorch.callbacks import EarlyStopping
from utils.ops import evaluate_results, load_ml_image, load_sb_image
import config, paths
import fire
import mlflow
from mlflow.pytorch import autolog

default = config.default
experiments = config.experiments
def train(
    exp_name,
    ):
    
    experiment = experiments[exp_name]
    
    run_name = experiment['run_name']
    model_name = experiment['model_name']
    experiment_name = experiment['experiment_name']
    model = experiment['model']
    criterion = experiment['criterion']
    optimizer = experiment['optimizer']
    data_module = experiment['data_module']
    train_params = experiment['train_params']
    
    model = model['class'](**model['params'])
    data_module = data_module['class'](**data_module['params'])
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
        limit_train_batches=200,
        limit_val_batches=200
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
        pred_results = save_pred_callback.final_image
        qd_paths_dict = paths.path_to_data['def_data']
        ref_data = load_ml_image(qd_paths_dict)[:,:,-22:]
        mask = load_sb_image(paths.path_to_mask)
        mse, mae, norm_mse, norm_mae = evaluate_results(ref_data, pred_results, mask, bins = [0, 1, 2, 5, 10, 30, 100])
        mlflow.log_metrics({
            'predicted_mse': mse,
            'predicted_mae': mae,
            'normalized_predicted_mse': norm_mse,
            'normalized_predicted_mae': norm_mae,
        })
    
  

if __name__ == '__main__':
    fire.Fire()