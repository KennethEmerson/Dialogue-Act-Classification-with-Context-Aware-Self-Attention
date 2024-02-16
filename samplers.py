#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains specific Pytorch samplers used by the dataloaders during training
#######################################################################################################

import numpy as np
import numpy.typing as npt
from torch.utils.data import Sampler, BatchSampler

###############################################################################################

class PerDialogueSampler(Sampler):
    """ Sampler that will create a uniform randomized permutation of all dialogue ids in the
        dataset and will iterate over this permutation. This way the order of dialogues during
        training is randomized.
    """
    
    def __init__(self, dialogue_ids:npt.NDArray):
        """initialize sampler
        Args:
            dialogue_ids (np.ndarray[str]): numpy array of dialogue_ids for each utterance in the dataset
        """
        self.dialogue_ids = dialogue_ids
        self.unique_dialogue_ids = np.unique(self.dialogue_ids)


    def __iter__(self):
        """creates uniform randomized permutation of the dialogue ids and then provides
           an iterator to iterate over the permutation
        Yields:
            _type_: the next dialogue ids in the permutation
        """
        permutation = np.random.permutation(self.unique_dialogue_ids) 
        for i in permutation:
            yield np.where(self.dialogue_ids == i)[0]


    def __len__(self) -> int:
        """returns the total amount of dialogues in the dataset
        Returns:
            int: total amount of dialogues in the dataset
        """
        return len(self.dialogue_ids)


############################################################################################### 

class PerDialogueBatchSampler(BatchSampler):
    """ Sampler that will create batches of dialogues. The batches are filled with utterances of the
        given dialogues (provided by the PerDialogueSampler) until the set batch size is reached. 
        This way utterances of one dialogue will always remain in order and in one batch 
    """
   
    def __init__(self, dialogue_ids:npt.NDArray, batch_size:int):
        """initialize sampler
        Args:
            dialogue_ids (np.ndarray[str]): numpy array of dialogue_ids in the dataset
            batch_size (int): the desired batch size. batches will be filled with dialogue utterances until size is reached
        """
        self.sampler = PerDialogueSampler(dialogue_ids)
        self.batch_size = batch_size


    def __iter__(self):
        """creates a batch with utterances of given dialogues, keeping the utterances per dialogue in order.
            Once batch size is reached, the batch is provided by the iterator.
            Due to the fact that utterances of one dialogue will always be kept in one batch, batch sizes may vary.
            The last batch will always contain the remaining dialogues and can thus be smaller than the requested batch size.
        Yields:
            np.array: a batch of utterances 
        """
        batch = []
        idx_in_batch = 0
        for idx in self.sampler:
            batch.append(idx)
            idx_in_batch += idx.size
            # only yield batch if size is reached
            if idx_in_batch >= self.batch_size:
                yield np.concatenate(batch)
                idx_in_batch = 0
                batch = []
        if idx_in_batch > 0:
            yield np.concatenate(batch)

    def __len__(self) -> int:
        """returns the total amount of batches in the dataset
        Returns:
            int: total amount of batches in the dataset
        """
        return len(self.sampler) // self.batch_size + 1