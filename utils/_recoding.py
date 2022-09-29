from pickletools import int4
from ._wandb import wandb_config

from torch.nn.modules import Module
from torch.optim import Optimizer
from torch import Tensor
import torch.nn as nn
import torch

class Recoder() :
    def __init__(self,
        args,
        model : Module,
        is_training : bool = True,
        scope : str = 'maxacc'
    ) :
        self.is_training = is_training
        self.model = model
        self.args = args
        self.scope = scope
        self.max_val_acc : float = 0.
        self.max_train_acc : float = 0.
        self.min_val_loss : float = float('inf')
        
        if args.wandb :
            self.wandb = wandb_config(
                args = args,
                criterion = model.criterion,
                optimizer = model.optimizer,
                is_training = is_training,
                scheduler = model.scheduler
            )

    def update(self,
        **kwargs
    ) :
        if self.is_training :
            self._update(**kwargs)
        else :
            self._update_test(**kwargs)

    def _update(self,
        train_acc : float,
        train_loss : float,
        train_time : float,
        val_acc : float,
        val_loss : float,
        epoch : int
    ):

        if self.max_train_acc < train_acc :
            self.max_train_acc = train_acc
        if self.max_val_acc < val_acc :
            self.max_val_acc = val_acc 
            if self.scope == 'maxacc' :
                self.model.model_save(isbase = True) 
        if self.min_val_loss > val_loss :
            self.min_val_loss = val_loss
            if self.scope == 'minloss' :
                self.model.model_save(isbase = False)

        print(
            f"[{epoch+1}/{self.args.epochs}]",
            f"train acc = {train_acc*100:.2f}%,",
            f"val acc = {val_acc*100:.2f}%,", 
            f"time = {train_time:.1f}s",
            f"Max train acc = {self.max_train_acc*100:.2f}%,",
            f"Max val acc = {self.max_val_acc*100:.2f}%,", 
        )

        if self.args.wandb :
            self.wandb.log( {'max train acc':self.max_train_acc }, step=epoch )
            self.wandb.log( {'train acc': train_acc}, step=epoch )
            self.wandb.log( {'max val acc': self.max_val_acc }, step=epoch )
            self.wandb.log( {'val acc': val_acc}, step=epoch )
            self.wandb.log( {'train loss': train_loss}, step=epoch )
            self.wandb.log( {'val loss': val_loss}, step=epoch )
            self.wandb.log( {'time': train_time}, step=epoch )

    def _update_test(
        self,
        method : str,
        epsilon : float,
        accuracy : float,
        success_rate : float,
        val_time : float
    ):
        print(
            f"[{method} / epsilon : {epsilon}]",
            f"accuracy = {accuracy*100:.2f}%,",
            f"success_rate = {success_rate*100:.2f}%,", 
            f"time = {val_time:.1f}s",
        )
        if self.args.wandb :
            self.wandb.log( {method + ' accuracy': accuracy }, step=epsilon )
            self.wandb.log( {method + ' success rate': success_rate}, step=epsilon )
            self.wandb.log( {method + ' val time': val_time}, step=epsilon )