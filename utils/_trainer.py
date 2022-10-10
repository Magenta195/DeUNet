from typing import List, Optional, Tuple, Union
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, device
from torch.utils.data import DataLoader
from torch.nn.modules import Module
from torch.optim import Optimizer

from ._metric import accuracy, attack_success_rate

from time import perf_counter

def _train(
    cur_epoch: int,
    total_epoch: int,
    model: Module,
    attack : None,
    dataloader: DataLoader,
    dev: device,
) -> Tuple[float, float, float]:
    train_loss = 0
    train_acc = 0
    total_size = 0

    model.train()
    start_time = perf_counter()
    mid_time = 0
    model.optimizer.zero_grad()
    for i, batch in enumerate( tqdm( dataloader, leave=True) ):
        # Training
        inputs: Tensor = batch[0].to(dev)
        labels: Tensor = batch[1].to(dev)
        
        tmp_time = perf_counter()
        orig_inputs = inputs.data
        if attack is not None:
            inputs = attack( model.base_model, inputs,  labels )
        mid_time += perf_counter() - tmp_time

        outputs, loss = model.cal_loss(
            orig_inputs = orig_inputs,
            inputs = inputs,
            labels = labels
        )
        

        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        # Obtain the total accuracy and loss (top-1)
        _, preds = torch.max(outputs.data, 1)
        train_loss += loss.item()
        train_acc += torch.sum( preds == labels.data ).item()
        total_size += labels.size(0)
        print( f"[{cur_epoch+1}/{total_epoch}][{i+1}/{len(dataloader)}]", end='\r' )

    end_time = perf_counter()

    training_time = end_time - start_time
    if model.is_filtered() : 
        training_time -= mid_time

    train_loss /= total_size
    train_acc /= total_size
    return train_loss, train_acc, training_time

def _validation(
    model: Module,
    dataloader: DataLoader,
    dev: device,
) -> Tuple[float, float, float]:
    val_loss = 0
    val_acc = 0
    total_size = 0

    model.eval()
    start_time = perf_counter()
    with torch.no_grad():
        for i, batch in enumerate( tqdm( dataloader, leave=True ) ):
            # Training
            inputs: Tensor = batch[0].to(dev)
            labels: Tensor = batch[1].to(dev)

            orig_inputs = inputs.data
            outputs, loss = model.cal_loss(
                orig_inputs = orig_inputs,
                inputs = inputs,
                labels = labels
            )
            # Obtain the total accuracy and loss (top-1)
            _, preds = torch.max(outputs.data, 1)
            val_loss += loss.item()
            val_acc += torch.sum( preds == labels.data ).item()
            total_size += labels.size(0)

    end_time = perf_counter()
    val_time = end_time - start_time
    val_loss /= total_size
    val_acc /= total_size
    return val_loss, val_acc, val_time

def robust_test(
    model: Module,
    attack : None,
    dataloader: DataLoader,
    dev: device,
) -> Tuple[float, float, float]:

    if attack is None :
        _, val_acc, val_time = _validation(
            model = model,
            dataloader = dataloader,
            dev = dev
        )
        return 0., val_acc, val_time 

    val_acc = 0
    val_atk_success = 0
    total_success_size = 0
    total_size = 0

    model.eval()
    start_time = perf_counter()
    for i, batch in enumerate( tqdm( dataloader )):
        inputs: Tensor = batch[0].to(dev)
        labels: Tensor = batch[1].to(dev)

        # adversarial_inputs = attack( model.base_model, inputs,  labels )
        adversarial_inputs = attack( model.base_model, inputs,  labels )

        outputs: Tensor = model.base_model( inputs )
        attacked_outputs: Tensor = model.base_model( adversarial_inputs )

        _, preds = torch.max(outputs.data, 1)
        _, adv_preds = torch.max(attacked_outputs.data, 1)

        filtered_adv_preds = None
        if model.is_filtered() :
            filtered_attacked_outputs: Tensor = model( adversarial_inputs )
            _, filtered_adv_preds = torch.max(filtered_attacked_outputs.data, 1)


        success_size, attack_success_size = attack_success_rate(
            prediction = preds,
            label = labels,
            attacked_prediction = adv_preds,
            filtered_prediction = filtered_adv_preds
        )
        if filtered_adv_preds is None :
            accuracy_size = accuracy( adv_preds, labels )
        else :
            accuracy_size = accuracy( filtered_adv_preds, labels )

        val_acc += accuracy_size.item()
        val_atk_success += attack_success_size.item()
        total_size += labels.size(0)
        total_success_size += success_size.item()

    end_time = perf_counter()
    val_time = end_time - start_time
    val_atk_success /= total_success_size
    val_acc /= total_size
    return val_atk_success, val_acc, val_time


def train_one_epoch(
    cur_epoch: int,
    total_epoch: int,
    attack : None,
    model: Module,
    trainloader: DataLoader,
    testloader: DataLoader, 
    dev: device,
) -> Tuple[float, float, float, float, float]:

    train_loss, train_acc, train_time = _train(
        cur_epoch,
        total_epoch,
        model,
        attack,
        trainloader,
        dev,
    )
    
    val_loss, val_acc, val_time = _validation(
        model,
        testloader,
        dev 
    )

    if model.scheduler is not None :
        model.scheduler.step()

    return train_acc, train_loss, val_acc, val_loss, train_time
