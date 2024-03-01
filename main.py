from models.models import ModelModule
import segmentation_models_pytorch as smp
import torch
from utils.datasets import ImageDataModule
from utils.callbacks import SavePrediction
from lightning.pytorch.trainer.trainer import  Trainer
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def train(exp_n):
    n_prev = 2
    
    #Data setup
    data_module = ImageDataModule(
        n_previous_times = n_prev,
        train_times = 34,
        val_times = 24,
        test_times = 24,
        patch_size = 32,
        train_overlap = 0.8,
        train_batch_size = 32,
        train_num_workers = 8,
        prediction_overlap = 0.5,
        features = ['ArCS', 'HIDR', 'Monthly']
    )
    
    #Model setup
    model = smp.UnetPlusPlus(
        #encoder_name="resnext101_32x8d",
        encoder_name="resnet34",
        in_channels=n_prev,
        classes = 1,
        encoder_weights = None,
        activation='identity'
    )
    criterion = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam
    lr = 2e-5
    modelModule = ModelModule(model, criterion, optimizer, lr)
    
    #Trainer setup
    mlf_logger = MLFlowLogger(
        experiment_name="Deforestation Prediction", 
        tracking_uri="file:./experiments/ml-runs",
        run_name=f"exp_{exp_n}",
        #log_model = True
        )
    
    callbacks = [
        EarlyStopping(
            monitor = 'val_loss_epoch',
            patience = 10,
            verbose = True
        ),
        ModelCheckpoint(
            dirpath = f'experiments/exp_{exp_n}',
            filename = 'model',
            monitor = 'val_loss_epoch',
            verbose = True
        ),
        SavePrediction(
            tiff_path = f'experiments/exp_{exp_n}/prediction.tif',
            n_prev = n_prev,
            patch_size = 32,
            test_times = 24,
            border_removal=4
        )
    ]
    trainer = Trainer(
        logger = mlf_logger,
        callbacks = callbacks,
        #max_epochs=5
    )
    
    #training
    trainer.fit(
        model = modelModule,
        datamodule = data_module
    )
    
    modelModule = ModelModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    torch.save(modelModule.model, f'experiments/exp_{exp_n}/model.pth')
    
    #Test
    trainer.test(
        model = modelModule,
        datamodule = data_module
    )
    
    #Prediction
    trainer.predict(
        model = modelModule,
        datamodule = data_module,
        return_predictions = False
    )
    

if __name__ == '__main__':
    train(0)