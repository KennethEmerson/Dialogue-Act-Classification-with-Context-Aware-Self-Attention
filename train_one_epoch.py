#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains function to train a model during one epoch
#######################################################################################################

from typing import Tuple
import time
import torch
from torch.utils.data import DataLoader

from model import ContextAwareDAC

def train_one_epoch(train_dataloader:DataLoader,
                    validation_dataloader:DataLoader, 
                    model:ContextAwareDAC, 
                    loss_fn,
                    device:torch.device,
                    gradient_clipping:float) -> Tuple[float,float]:
    """trains the model during one epoch

    Args:
        train_dataloader (DataLoader): the dataloader providing the batches of training data
        validation_dataloader (DataLoader): the dataloader providing the batches of validation data
        model (ContextAwareDAC): the actual model to train
        loss_fn (_type_): the Pytorch loss function to use
        device (torch.device): the Pytorch device to use to run the model (e.g. CUDA,mps,cpu)
        gradient_clipping (float): the gradient norm treshold to clip

    Raises:
        Exception: is raised if the model has no optimizer initialised

    Returns:
        Tuple[float,float]: the average training loss and average validation loss
    """
    
    # check is the model contains an optimizer object
    if not isinstance(model.optimizer,torch.optim.Optimizer):
        raise Exception("model optimizer is not set")
    
    # train
    model.train()
    running_training_loss = 0.0
    batch_count = 0
    for data_batch, intents, targets in train_dataloader:
        batch_start_time = time.process_time()
        
        model.zero_grad()
        predictions = model(data_batch)
        loss = loss_fn(predictions, targets.to(device))
        running_training_loss += loss.item()
        loss.backward()
        
        grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
        norm = torch.cat(grads).norm()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping) # type: ignore

        model.optimizer.step()
        elapsed_time = time.process_time() - batch_start_time
        print(f"training batch: {batch_count} batchsize: {len(data_batch)} gradient norm: {norm:.5f} duration[s]: {elapsed_time:.0f}")
        batch_count += 1 

    avg_training_loss = running_training_loss / (batch_count + 1)        

    # validate
    running_validation_loss = 0.0
    validation_batch_count = 1
    model.eval()
    with torch.no_grad():
        for data_batch, intents, targets in validation_dataloader:
            batch_start_time = time.process_time()
            predictions = model(data_batch)
            validation_loss = loss_fn(predictions, targets.to(device))
            running_validation_loss += validation_loss.item()
            elapsed_time = time.process_time() - batch_start_time
            print(f"validation batch: {validation_batch_count} batchsize: {len(data_batch)} duration[s]: {elapsed_time:.0f}")
            validation_batch_count += 1

    avg_validation_loss = running_validation_loss / (validation_batch_count + 1)
    
    return avg_training_loss, avg_validation_loss