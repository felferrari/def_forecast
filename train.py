from utils.datasets import TrainDataset
from pathlib import Path
from models.models import Model1
from torch.utils.data import DataLoader
import lightning as L


mask_path = r'data/tiff/mask.tif'
ArCS_path = r'data/tiff/ArCS.tif'


train_ds = TrainDataset(ArCS_path, mask_path, 6, [0, 34])
val_ds = TrainDataset(ArCS_path, mask_path, 6, [34, 68])
train_loader = DataLoader(train_ds, batch_size=16, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=16, num_workers=1)

model = Model1(6)

trainer = L.Trainer(
    log_every_n_steps = 1,
    enable_progress_bar = True
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)