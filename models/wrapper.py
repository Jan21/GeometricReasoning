import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from models.model import NeuroSAT

class Pl_model_wrapper(pl.LightningModule):
    def __init__(self, 
                 model_cfg,
                 train_cfg,
                 data_cfg,
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
            n_vote_layers=model_cfg.get('n_vote_layers', 0),
            max_size=data_cfg.max_size
        )
        
        self.loss_fn = loss_fn
        self._epochs = train_cfg.num_epochs  
        self.scheduler = train_cfg.scheduler 
        self.num_iters = model_cfg.num_iters

    def forward(self, batch):
        return self.model(batch, self.num_iters)

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat, _ = self(batch)
        vars_idx = torch.nonzero(1-batch.p_type).view(-1)
        y_hat = y_hat[vars_idx]
        y = y[vars_idx]
        
        _, pred = y_hat.max(dim=1)
        num_correct = pred.eq(y).sum().item()
        num_total = len(y)
        acc = num_correct / num_total
        
        loss = self.loss_fn(y_hat, y)
        
        # Log more metrics
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)
        self.log('batch_size', float(batch.num_clauses.shape[0]), prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y
        y_hat, _ = self(batch)
        
        # Get indices of unknown points
        vars_idx = torch.nonzero(1-batch.p_type).view(-1)
        y_hat = y_hat[vars_idx]
        y = y[vars_idx]
        
        # Original per-point accuracy calculation
        _, pred = y_hat.max(dim=1)
        num_correct = pred.eq(y).sum().item()
        num_total = len(y)
        acc = num_correct / num_total
        
        # New complete problem accuracy calculation
        complete_correct = 0
        total_problems = 0
        
        # Get unique batch indices for unknown points
        batch_nums = batch.x_p_batch[vars_idx]
        unique_batches = torch.unique(batch_nums)
        
        # Check each problem separately
        for b in unique_batches:
            # Get predictions and ground truth for current problem's unknown points
            problem_mask = batch_nums == b
            problem_pred = pred[problem_mask]
            problem_y = y[problem_mask]
            
            # Check if all predictions match for this problem
            if torch.all(problem_pred == problem_y):
                complete_correct += 1
            total_problems += 1
        
        complete_acc = complete_correct / total_problems if total_problems > 0 else 0
        
        loss = self.loss_fn(y_hat, y)
        
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_clauses.shape[0])
        self.log('val_complete_acc', complete_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_clauses.shape[0])
        self.log('val_loss', loss, batch_size=batch.num_clauses.shape[0])
        
        return loss
    
    def test_step(self, batch, batch_idx):
        y = batch.y
        y_hat, _ = self(batch)
        
        # Get indices of unknown points
        vars_idx = torch.nonzero(1-batch.p_type).view(-1)
        y_hat = y_hat[vars_idx]
        y = y[vars_idx]
        
        # Per-point accuracy calculation
        _, pred = y_hat.max(dim=1)
        num_correct = pred.eq(y).sum().item()
        num_total = len(y)
        acc = num_correct / num_total
        
        # Complete problem accuracy calculation
        complete_correct = 0
        total_problems = 0
        
        # Get unique batch indices for unknown points
        batch_nums = batch.x_p_batch[vars_idx]
        unique_batches = torch.unique(batch_nums)
        
        # Check each problem separately
        for b in unique_batches:
            problem_mask = batch_nums == b
            problem_pred = pred[problem_mask]
            problem_y = y[problem_mask]
            
            if torch.all(problem_pred == problem_y):
                complete_correct += 1
            total_problems += 1
        
        complete_acc = complete_correct / total_problems if total_problems > 0 else 0
        
        loss = self.loss_fn(y_hat, y)
        
        # Log all metrics
        self.log('test_acc', acc, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_clauses.shape[0])
        self.log('test_complete_acc', complete_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_clauses.shape[0])
        self.log('test_loss', loss, batch_size=batch.num_clauses.shape[0])
        
        return loss

    def configure_optimizers(self):
        if self.scheduler == "cosine":
            print("cosine scheduler used")
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._epochs)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif self.scheduler == "cosinerestart":
            print("cosine with restarts scheduler used")
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4, 2)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif self.scheduler == "linear":
            print("linear scheduler used")
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 1e-4, total_iters=self._epochs)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}