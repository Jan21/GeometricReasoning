import torch
import pytorch_lightning as pl
from data.geo_data import get_Geo_dataset, Geo_datamodule
from models.wrapper import Pl_model_wrapper
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model_signature = f"GeoReasoning_{cfg.model.name}"
    logging.info(f"Model signature: {model_signature}")
    
    dataset = get_Geo_dataset(
        cfg.data.data_path,
        cfg.data.max_size,
        cfg.data.max_constraints,
        cfg.data.valid_split
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()

    wrapped_model = Pl_model_wrapper(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        data_cfg=cfg.data,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        loss_fn=loss_fn
    )
    
    logger = WandbLogger(project="GeoReasoning", name=model_signature)
    
    data = Geo_datamodule(dataset, cfg.data.batch_size)
    
    # Create timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Combine model signature and timestamp into one folder name
    checkpoint_dir = f"{cfg.system.base_dir}/checkpoints/{model_signature}_{timestamp}"
    
    callbacks = [LearningRateMonitor(logging_interval='step')]
    
    # Add checkpoint callback if enabled in config
    if cfg.train.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:03d}-{val_acc:.3f}",
            save_top_k=-1,  # Save all checkpoints
            every_n_epochs=10,  # Save every epoch
            save_on_train_epoch_end=False,  # Save on validation end
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=logger,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks
    )
    
    trainer.fit(wrapped_model, data)
    trainer.save_checkpoint(f"{model_signature}.ckpt")

if __name__ == '__main__':
    main()