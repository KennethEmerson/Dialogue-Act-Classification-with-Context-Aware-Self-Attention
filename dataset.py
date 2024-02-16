#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains Pytorch Dataset class used to feed the PyTorch DataLoaders during training and
# evaluation.
#######################################################################################################
from typing import Dict, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np
from pandera.typing import DataFrame

from dataframe_schema import DataSchema


class DADataset(Dataset):
    
    def __init__(self, data:DataFrame[DataSchema],label_dict:Dict[str,int]) -> None:
        """ intialize the Torch Dataset
        Args:
            data (DataFrame[DataSchema]): the Pandas dataframe containing the dataset
            label_dict (Dict[str,int]): the Label dictionary mapping the utterance intents/ class labels to a class index
        """
        self.utterances = np.array(data["utterance"]) #.to_numpy()
        self.dialogue_ids = np.array(data["dialogue_id"]) 
        self.intents = np.array(data["intent"]) 
        self.label_dict = label_dict


    def __len__(self) -> int:
        """ returns the total amount of all utterances in the dataset
        Returns:
            int: amount of utterances in the dataset
        """
        return len(self.utterances)
    

    def __getitem__(self, index:int) -> Tuple[str,str,torch.Tensor]:
        """ returns one utterance, its intent and the intent index based on the utterance index in the dataset
        Args:
            index (int): the index of the utterance to retrieve
        Returns:
            Tuple[str,str,int]: the utterance, its intent and the intent index
        """
        utterance = self.utterances[index]
        intent = self.intents[index]
        intent_index = torch.tensor(self.label_dict[intent], dtype=torch.long)
        
        return utterance, intent, intent_index



    
