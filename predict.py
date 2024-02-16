#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains function and main code to predict utterance intents using a saved model.
# Prior to using the predict function, a model must be stored.
#######################################################################################################

from typing import Union, List, Dict
import mlflow
import pandas as pd
import torch
import numpy as np
from pandera.typing import DataFrame

import data.load_data as data_loader
from dataframe_schema import DataSchema

###############################################################################################

def predict_dialogue_from_df(
        data:DataFrame[DataSchema],
        label_list:List[str],
        labels:Dict[str,int],
        mlflow_model_name:str,
        mlflow_model_version:int,
        dialogue_id:Union[str,list[str]]) -> pd.DataFrame:
    """ makes a prediction of one or more complete dialogues as stored in the dataframe

    Args:
        data (DataFrame[DataSchema]): the dataframe containing the utterance and intents
        label_list (List[str]): the list with intent strings to use a class labels. Must be identical to the one used during training.
        labels (Dict[str,int]): dictionary mapping of intent strings/class labels to class indexes. Must be identical to the one used during training.
        mlflow_model_name (str): the model name as saved in MLFLow
        mlflow_model_version (int): the model version as saved in MLFLow
        dialogue_id (Union[str,list[str]]): the dialogue_id or list of dialogue_ids for which to predict the utterance intents

    Raises:
        Exception: if the dialogue_id can not be found, an exception will be raised

    Returns:
        pd.DataFrame: the original dataframe augmented with the predicted intents and the prediction distribution
    """
    
    # transform to list if only one dialogue_id is provided
    dialogue_id = ([dialogue_id] if isinstance(dialogue_id,str) else dialogue_id)
    
    # get conversations for the given dialogue_ids.
    data_filtered = data[data["dialogue_id"].isin(dialogue_id)].copy()
    if len(data) == 0:
        raise Exception("dialogue_id does not exist")
    utterances = list(data_filtered["utterance"])

    # load the model
    model_uri = f"models:/{mlflow_model_name}/{mlflow_model_version}"
    model = mlflow.pytorch.load_model(model_uri)
    
    # make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(utterances)
    softmax_predictions = torch.nn.functional.softmax(predictions,dim=1)
    data_filtered["prediction"] = [label_list[i] for i in torch.argmax(softmax_predictions, 1).tolist()]
    return data_filtered
    
        
###############################################################################################

if __name__=="__main__":
    
    DATA_FOLDER = "data"
    
    # model name and version as stored in MLFLow
    MODEL_NAME = "CASA-dialogue-ACT-classifier-V1.0"
    MODEL_VERSION = 6
    # dialogue_id as known in dataset
    dialogue_id="1_00043"

    data = data_loader.get_training_data(DATA_FOLDER)
    label_list = data_loader.get_label_list(DATA_FOLDER)
    labels = data_loader.get_labels(DATA_FOLDER)
    
    results = predict_dialogue_from_df(
        data=DataFrame[DataSchema](data),
        label_list = label_list,
        labels = labels,
        mlflow_model_name = MODEL_NAME,
        mlflow_model_version = MODEL_VERSION,
        dialogue_id=dialogue_id)
    
    print(results.drop(columns=["dialogue_id","frame"]))