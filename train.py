import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wrapper import Pl_model_wrapper
from models import models_with_args
import numpy as np
import random
import time
import pickle
from data.geo_data import get_Geo_dataset, Geo_datamodule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


def nonincremental_train(model,
                   dataset,
                   batch_size,
                   max_epochs,
                   grad_clip,
                   num_iters,
                   logger,
                   checkpoint,
                   scheduler):

    model.num_iters = num_iters
    model._epochs = max_epochs
    model.scheduler = scheduler
    data = Geo_datamodule(dataset, batch_size)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         logger=logger,
                         accelerator="gpu", devices=1,
                         gradient_clip_val=grad_clip,callbacks=[lr_monitor])
    trainer.fit(model, data)
    trainer.save_checkpoint(checkpoint)
    return model
                             
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('-bs', '--batch_size', type=int, default=64) #64, 256, 1024
    parser.add_argument('-itrs', '--num_iters', type=int, default=25) # 10, 20, 30
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3) 
    parser.add_argument('-schd', '--scheduler', type=str, default="cosine") #linear, cosine, cosinerestart
    parser.add_argument('-epoch', '--max_epoch', type=int, default=200)
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # set hyperparameters
    lr = args.learning_rate
    weight_decay = 1e-10
    model_name = 'NeuroSAT'
    logger = TensorBoardLogger("temp/tb_logs", name="Final",)

    checkpoint = "final_model_checkpoint.ckpt" #args.checkpoint 
    data_path = 'temp/geo/geometry_problems_most_new.pickle' #args.datapath
    
    #incremental = args.incremental
    batch_size = args.batch_size
    gpus = [0]
    grad_clip = 0.65
    num_iters = args.num_iters
    max_epochs = args.max_epoch
    
    # create dataset and model
    dataset = get_Geo_dataset(data_path)
    model_class = models_with_args[model_name]
    loss_fn =  nn.CrossEntropyLoss()
    
    model = Pl_model_wrapper(model_class, lr, weight_decay, loss_fn)
    
    #model = Pl_model_wrapper.load_from_checkpoint("temp/new_epoch=0-step=80_82.ckpt")
                           

    model = nonincremental_train(model,
                            dataset,
                           batch_size,
                            max_epochs,
                            grad_clip,
                            num_iters,
                            logger,
                            checkpoint,
                            scheduler=args.scheduler)
     
