#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains function and main code to train a model ro to continue training an existing model
#######################################################################################################

from typing import Optional, List, Dict
import time
import datetime
import math
import random

import torch
from torch.utils.data import DataLoader
import mlflow
import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from dataset import DADataset
from dataframe_schema import DataSchema
from train_one_epoch import train_one_epoch
from model import ContextAwareDAC 
from samplers import PerDialogueBatchSampler
from model import *


# Constants
MLFLOW_EXPERIMENT_NAME = "CASA-dialogue-ACT-classifier"
MODEL_VERSION = "V1.0"
MAX_UTTERANCE_LENGTH = 87
MODEL_HIDDEN_SIZE = 768
DATASET_NBR_OF_INTENTS = 40
LEARNING_RATE = 0.015
DATALOADER_WORKERS = 2
GRADIENT_CLIPPING = 5.0

# Set MLFLOW
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


###############################################################################################

def train_model(
        train_data:DataFrame[DataSchema],
        validation_data:DataFrame[DataSchema],
        label_list: List[str],
        labels:Dict[str,int],
        epochs:int,
        batchsize:int, 
        mlflow_model_name:Optional[str]=None,
        mlflow_model_version:Optional[int]=None,
        run_id:Optional[str]=None,
        save_model:bool=False,
        early_stopping_limit = 5,
        seed:Optional[int]= None
        ):
    """Train a new or existing model

    Args:
        train_data (pd.DataFrame): the training data to use
        validation_data (pd.DataFrame): the validation data to use
        label_list (List[str]): the list with intent strings to use a class labels
        labels (Dict[str,int]): dictionary mapping of intent strings/class labels to class indexes.
        epochs (int): number of epochs to train, must be a positive number.
        batchsize (int): approx batchsize to use in each epoch
        mlflow_model_name (Optional[str], optional): if training of an existing model is resumed, provide its MLFlow name. Defaults to None.
        mlflow_model_version (Optional[int], optional): if training of an existing model is resumed, provide its MLFlow version. Defaults to None.
        run_id (Optional[str], optional): if training of an existing model is resumed, provide the MLFlow experiment run_id. Defaults to None.
        save_model (bool, optional): if True, model will be saved in MLFlow. Defaults to False.
        early_stopping_limit (int, optional): indicates how many epochs, validation loss may increase before training is stopped. Defaults to 5.
        seed (int,optional): if provided use this seed to initialize new model. will be ignored when loading an existing model

    Raises:
        Warning: a warning is logged in the terminal if utterances in the training or vaildation data exceed the MAX_UTTERANCE_LENGTH constant value
    """

    # check for input consistency
    if(epochs<1):
         raise Exception("[ERROR]: number of epochs must be bigger than zero")
    if(batchsize<1):
         raise Exception("[ERROR]: batchsize must be bigger than zero")
    if(mlflow_model_name and not mlflow_model_version):
         raise Exception("[ERROR]: when providing a model name, a model version should also be provided")
    if(not mlflow_model_name and mlflow_model_version):
         raise Exception("[ERROR]: when providing a model version, a model name should also be provided")
    if(mlflow_model_name and mlflow_model_version and seed):
         raise Warning("[WARNING]: The custom seed will be ignored when using an existing model")
    if(run_id and (not mlflow_model_name or not mlflow_model_version)):
         raise Warning("[WARNING]: when providing a run_id, it does not make sense to not add the name/version of the model you wish to resume training")

    # data validation
    if(train_data["utterance"].str.split(" ").str.len().max() > MAX_UTTERANCE_LENGTH or
        validation_data["utterance"].str.split(" ").str.len().max() > MAX_UTTERANCE_LENGTH):
        raise Warning("[WARNING]: real utterance length is bigger than MAX_UTTERANCE_LENGTH, utterances will be truncated")


    # detect if GPU is present otherwise use cpu
    # device_name = (
    #         "cuda"
    #         if torch.cuda.is_available() and torch.backends.cuda.is_built()
    #         else "mps"
    #         if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    #         else "cpu"
    #         )   
    # device = torch.device(device_name)
    
    device_name = "cpu"
    device = torch.device("cpu")


    minimal_val_loss = math.inf
    early_stopping_counter = 0


    with mlflow.start_run(run_name=f"{MLFLOW_EXPERIMENT_NAME}-{MODEL_VERSION}",run_id=run_id if run_id else ""):
        
        # get the initial epoch number if continuing an existing experiment run
        # (required to set the metrics to a specific epcoh step in MLFlow)
        if run_id:
            start_epoch = len(mlflow.MlflowClient().get_metric_history(run_id,"validation loss"))     
        else:
            start_epoch = 0
        end_epoch = start_epoch + epochs


        # load existing model if required els create new model
        if mlflow_model_name and mlflow_model_version:
            model_uri = f"models:/{mlflow_model_name}/{mlflow_model_version}"
            model = mlflow.pytorch.load_model(model_uri)
            model.reset_device(device)
            model.to(device)
        else:    
            if not seed:
                seed = random.randint(0,100000)
            random.seed(seed)
            torch.manual_seed(seed)
            model = ContextAwareDAC(
                    labels=labels,
                    hidden_size = MODEL_HIDDEN_SIZE, # unused parameter, could be used to set hidden states in model
                    max_tokens_per_utternace= MAX_UTTERANCE_LENGTH,
                    device=device
                ).to(device)
        

        # set optimizer and loss function
        model.set_optimizer(torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE,amsgrad=True))
        loss_fn = torch.nn.functional.cross_entropy
 

        # print start of training
        print("-"*100)
        print(f"start training with device: {device_name} and seed: {seed}")
        print(f"training set size: {len(train_data)}, validation dataset size: {len(validation_data)}")
        print("-"*100)
        

        # create Pytorch datasets, samplers and dataloaders
        train_dataset = DADataset(train_data,label_dict=labels)
        train_sampler = PerDialogueBatchSampler(train_dataset.dialogue_ids,batch_size=batchsize)
        train_loader = DataLoader(dataset=train_dataset, num_workers=DATALOADER_WORKERS, batch_sampler=train_sampler)
        validation_dataset = DADataset(validation_data,label_dict=labels)
        validation_sampler = PerDialogueBatchSampler(validation_dataset.dialogue_ids,batch_size=batchsize)
        validation_loader = DataLoader(dataset=validation_dataset, num_workers=DATALOADER_WORKERS, batch_sampler=validation_sampler)


        # log MLFlow parameters
        mlflow.log_param("computation device",device_name)
        mlflow.log_param("model hidden size",MODEL_HIDDEN_SIZE)
        mlflow.log_param("dataset nbr of intents",DATASET_NBR_OF_INTENTS)
        mlflow.log_param("optimizer_learning_rate",LEARNING_RATE)
        mlflow.log_param("batchsize",batchsize)
        if not run_id:
            mlflow.log_param("seed",seed)
        mlflow.log_param("gradient clipping treshold",GRADIENT_CLIPPING)
        mlflow.log_param("ratio of training dataset used for validation",len(validation_data)/(len(validation_data)+len(train_data)))
        mlflow.log_param("number of records in training set",len(train_data)) 
        mlflow.log_param("number of records in validation set",len(validation_data))


        # Start training of epochs
        for epoch_count in range(start_epoch,end_epoch,1):
            
            epoch_start_time = time.process_time()
            avg_training_loss, avg_validation_loss = train_one_epoch(train_loader,validation_loader, model, loss_fn,device,GRADIENT_CLIPPING)
            elapsed_time = time.process_time() - epoch_start_time
            mlflow.log_metrics({"training loss":avg_training_loss,"validation loss":avg_validation_loss,"duration":elapsed_time},step=epoch_count)
            print("-"*100)
            print(f"Epoch: {epoch_count} training loss: {avg_training_loss:.6f} validation loss: {avg_validation_loss:.6f} duration: {str(datetime.timedelta(seconds=elapsed_time))}")
            print("-"*100)

            # adjust early stopping counter
            if avg_validation_loss < minimal_val_loss:
                minimal_val_loss = avg_validation_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter +=1
            
            # check if early stopping treshold is met and stop if appropriate
            if early_stopping_counter > early_stopping_limit:
                mlflow.log_metrics({"early stopping":1},step=epoch_count)
                break
        
        # save model using MLFlow if requested
        if save_model:
            model_info = mlflow.pytorch.log_model(model, "model",registered_model_name=f"{MLFLOW_EXPERIMENT_NAME}-{MODEL_VERSION}")


###############################################################################################


if __name__=="__main__":
    
    DATA_FOLDER = "data"
    DEBUG_SAMPLE_LIMIT = None # can be set to an integer value to limit dataset size for debugging
    EPOCHS = 20
    BATCHSIZE = 1400
    SAVE_MODEL = True
    EARLY_STOPPING = 9
    

    # Load data
    train_data = DataFrame[DataSchema](pd.read_csv(f"./{DATA_FOLDER}/train.csv",nrows=DEBUG_SAMPLE_LIMIT))
    validation_data = DataFrame[DataSchema](pd.read_csv(f"./{DATA_FOLDER}/validate.csv",nrows=DEBUG_SAMPLE_LIMIT))
    label_list = np.loadtxt("./" + DATA_FOLDER + "/labels.csv",dtype=str).tolist()
    labels = {k: v for v, k in enumerate(label_list)}

    # start training new model
    train_model(train_data=train_data,
                validation_data=validation_data,
                label_list=label_list,
                labels = labels,
                epochs=EPOCHS,
                batchsize=BATCHSIZE,
                save_model=SAVE_MODEL,
                mlflow_model_name = None,
                mlflow_model_version = None,
                run_id= None,
                early_stopping_limit= EARLY_STOPPING,
                seed=None
                )

    # resume training from existing model
    # train_model(train_data=train_data,
    #             validation_data=validation_data,
    #             label_list=label_list,
    #             labels = labels,
    #             epochs=EPOCHS,
    #             batchsize=BATCHSIZE,
    #             save_model=SAVE_MODEL,
    #             mlflow_model_name = "CASA-dialogue-ACT-classifier-V1.0",
    #             mlflow_model_version = 6,
    #             run_id= "382131c84d8a445cbe511620d3b7a27f",
    #             early_stopping_limit= EARLY_STOPPING,
    #             seed = None
    #             )