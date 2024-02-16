#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains functions to preprocess the original data in order to be used to train and evaluate
# the model
#######################################################################################################

from typing import Tuple, Dict, List
import sys
import os
import json
import math

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

sys.path.insert(1, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from dataframe_schema import DataSchema


def get_files(folder:str) -> list[str]:
    """returns a list of files in a folder
    Args:
        folder (str): path of folder for which a list of files is desired
    Returns:
        list[str]: list of files in the folder
    """
    onlyfiles = [os.path.join(folder,f) for f in os.listdir(folder) if (os.path.isfile(os.path.join(folder, f)))]
    return onlyfiles


def get_data_from_file(filename:str)->list[dict]:
    """read the data from the datafile and include an EOU token representing the end of the conversation
    Args:
        filename (str): the name of the json file containing the data
    Returns:
        list[dict]: a list of dict for which each dict represents an utterance data record
    """
    temp_data = []
    
    f = open( filename, "rb" )
    jsonObject = json.load(f)

    # iterate over conversations
    for dialogue in jsonObject:
        dialogue_id = dialogue["dialogue_id"]
        
        # load data for each utterance in a conversation
        for turn in dialogue["turns"]:
            intent = turn["dialogue_act"]
            utterance = turn['utterance']
            speaker = turn['speaker']
            temp_data.append({"dialogue_id":dialogue_id,"speaker":speaker,"frame":0,"intent":intent,"utterance":utterance})
        
        # add a special token on the end of each conversation
        temp_data.append({"dialogue_id":dialogue_id,"speaker":"EOU","frame":0,"intent":"EOU","utterance":"EOU EOU"})
    
    f.close()
    return temp_data


def load_data_from_folder_to_df(folder:str) -> DataFrame[DataSchema]:
    """load the data from all json files in the given folder and concatenate all data into one pandas dataframe
    Args:
        folder (str): the folder containing all the data files
    Returns:
        DataFrame[DataSchema]: the pandas dataframe containing all the data and conforming to the given DataSchema
    """
    onlyfiles = get_files(folder)
    data = []
    for file in onlyfiles:
        data += (get_data_from_file(file))

    return DataFrame[DataSchema](data)


def load_data_from_folder_and_split_to_train_and_val_df(folder:str,ratio:float)-> Tuple[DataFrame[DataSchema],DataFrame[DataSchema]]:
    """load the data from all json files in the given folder and concatenate all data into one pandas dataframe that is 
        then split into a training and validation set using the given utterance ratio. The split will always keep all utterances 
        of one conversation together, preventing utterances of one conversation being split into both datasets.
    Args:
        folder (str): the folder containing all the data files
        ratio (float): the ratio of utterance related split between training and validation
    Returns:
        Tuple[DataSchema,DataSchema]: the training and validation dataframes
    """
    onlyfiles = get_files(folder)
    
    data = []
    for file in onlyfiles:
        data += (get_data_from_file(file))
    
    data_df  = DataFrame[DataSchema](data)
    
    dialogue_counts = data_df['dialogue_id'].value_counts(sort=False)
    ocurrence_df = pd.DataFrame({'dialogue_id':dialogue_counts.index, 'count':dialogue_counts.values})
    ocurrence_df["cumsum"] = ocurrence_df["count"].cumsum()
    treshold = math.floor(len(data_df) * ratio)
    treshold_dialogue_id = ocurrence_df.iloc[(ocurrence_df['cumsum']-treshold).abs().argsort()[:1]]["dialogue_id"]

    split_index = data_df["dialogue_id"].values.searchsorted(treshold_dialogue_id, side='right')[0]
    train_data = data_df.iloc[:split_index]
    validation_data = data_df[split_index:]

    return DataFrame[DataSchema](train_data), DataFrame[DataSchema](validation_data)


def save_to_files(dataset_path:str,train_val_ratio:float) -> None:
    """loads all training data from the train and test folder in the given root folder, splits the training data into a training 
        and validation dataframe and stores them into two separate CSV files (train.csv and validate.csv). The test data is store 
        in a third test.csv file. Finally, a list of all intent labels is also saved in labels.csv

    Args:
        dataset_path (str): path to the root folder containing the train and test folder containing the respective data files
        train_val_ratio (float): _description_
    """
    
    train_file_path = dataset_path + "/train"
    test_file_path = dataset_path + "/test"

    train_data, validation_data = load_data_from_folder_and_split_to_train_and_val_df(train_file_path,train_val_ratio)
    test_data = load_data_from_folder_to_df(test_file_path)

    labels = (np.unique(np.concatenate((train_data["intent"].unique(),validation_data["intent"].unique(),test_data["intent"].unique()))))
    np.savetxt(dataset_path + "/labels.csv",labels,delimiter=",",fmt='%s')

    train_data.to_csv(dataset_path +  "/train.csv")
    validation_data.to_csv(dataset_path +"/validate.csv")
    test_data.to_csv(dataset_path + "/test.csv")
    print(f"data saved with train validation ratio: {train_val_ratio}")


def get_training_data(dataset_path:str) -> pd.DataFrame:
    """get the training data as a pd.DataFrame 
    Args:
        dataset_path (str): path where the train.csv can be found
    Returns:
        pd.DataFrame: containing the training data
    """
    filepath = os.path.join(dataset_path,"train.csv")
    return pd.read_csv(filepath)


def get_validation_data(dataset_path:str) -> pd.DataFrame:
    """get the validation data as a pd.DataFrame 
    Args:
        dataset_path (str): path where the validate.csv can be found
    Returns:
        pd.DataFrame: containing the validation data
    """
    filepath = os.path.join(dataset_path,"validate.csv")
    return pd.read_csv(filepath)


def get_test_data(dataset_path:str) -> pd.DataFrame:
    """get the test data as a pd.DataFrame 
    Args:
        dataset_path (str): path where the test.csv can be found
    Returns:
        pd.DataFrame: containing the test data
    """
    filepath = os.path.join(dataset_path,"test.csv")
    return pd.read_csv(filepath)


def get_labels(dataset_path:str) -> dict[str,int]:
    """get the labels  as a list
    Args:
        dataset_path (str): path where the labels.csv can be found
    Returns:
        List[str]: containing the labels
    """
    labels_list = get_label_list(dataset_path)
    return {k: v for v, k in enumerate(labels_list)}


def get_label_list(dataset_path:str) -> List[str]:
    """get the labels  as a dictionary mapping the labels to their index
    Args:
        dataset_path (str): path where the labels.csv can be found
    Returns:
       Dict[str,int]: containing the labels as keys and their index as values
    """
    filepath = os.path.join(dataset_path,"labels.csv")
    return np.loadtxt(filepath,dtype=str).tolist()



###################################################################################################
# Load, preprocess and store data into train.csv, validate.csv, test.csv and labels.csv
###################################################################################################
if __name__=="__main__":
    
    train_val_ratio = 0.9

    if len(sys.argv) > 1 and isinstance(sys.argv[1],float) and sys.argv[1] <= 1.:
        train_val_ratio = sys.argv[1]

    save_to_files(os.path.dirname(__file__),train_val_ratio)
   
    