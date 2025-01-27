import torch
import pytorch_lightning as pl
from data.geo_data import get_Geo_dataset, Geo_datamodule
from models.wrapper import Pl_model_wrapper
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import random

logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model_signature = f"GeoReasoning_{cfg.model.name}"
    logging.info(f"Model signature: {model_signature}")
    
    dataset = get_Geo_dataset(cfg.data.data_path)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    wrapped_model = Pl_model_wrapper(
        model_cfg=cfg.model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        loss_fn=loss_fn
    )
    
    logger = WandbLogger(project="GeoReasoning", name=model_signature)
    
    data = Geo_datamodule(dataset, cfg.data.batch_size)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=logger,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=[lr_monitor]
    )
    
    trainer.fit(wrapped_model, data)
    trainer.save_checkpoint(f"{model_signature}.ckpt")

if __name__ == '__main__':
    main()