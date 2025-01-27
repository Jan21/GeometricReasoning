import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from models.model import NeuroSAT

class Pl_model_wrapper(pl.LightningModule):
    def __init__(self, 
                 model_cfg,
                 lr, 
                 weight_decay,
                 loss_fn
                ):
        super(Pl_model_wrapper, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.model = NeuroSAT(
            d=model_cfg.d,
            final_reducer=model_cfg.final_reducer,
            n_msg_layers=model_cfg.get('n_msg_layers', 0),
            n_vote_layers=model_cfg.get('n_vote_layers', 0)
        )
        
        self.loss_fn = loss_fn
        self._epochs = 200
        self.scheduler = "linear"
        self.num_iters = model_cfg.num_iters

    def forward(self, batch):
        return self.model(batch, self.num_iters)

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat, _ = self(batch)
        vars_idx = torch.nonzero(1-batch.p_type).view(-1)
        y_hat = y_hat[vars_idx]
        y = y[vars_idx]
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y
        y_hat, _ = self(batch)
        vars_idx = torch.nonzero(1-batch.p_type).view(-1)
        y_hat = y_hat[vars_idx]
        y = y[vars_idx]
        # get argmax of y_hat
        _, pred = y_hat.max(dim=1)
        # get number of correct predictions
        num_correct = pred.eq(y).sum().item()
        num_total = len(y)
        acc = num_correct / num_total
        loss = self.loss_fn(y_hat, y)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_clauses.shape[0])
        self.log('val_loss', loss, batch_size=batch.num_clauses.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        y = batch.y
        y_hat, _ = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        pred = y_hat.max(dim=1)[1]
        acc = pred.eq(y).sum().item() / y.shape[0]
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        if self.scheduler == "cosine":
            print("cosine scheduler used")
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._epochs)
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-4, total_iters=5)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, lrs], milestones=[5])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif self.scheduler == "cosinerestart":
            print("cosine with restarts scheduler used")
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            lrs = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4, 2)
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-4, total_iters=5)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, lrs], milestones=[5])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif self.scheduler == "linear":
            print("linear scheduler used")
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            lrs = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 1e-4, total_iters=self._epochs//1.15)
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-4, total_iters=5)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, lrs], milestones=[5])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}