#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains Embedder class used by the model to embed the utterances and its characters
# based on paper referred to in original paper:
# Named entity recognition with bidirectional lstm-cnns by (Chiu and Nichols, 2016)
#######################################################################################################

from typing import Dict, List
import torch
from torchtext.data import get_tokenizer
import numpy as np
import string
import json
CHAR_EMBED_SIZE = 50

class Glove_and_Char_Embedder:
    saved_char_embedding_filepath = ''

    def __init__(self,glove_filepath:str,saved_filepath:str,max_tokens_per_utternace=87,max_char_per_token=21):
        """ intialize the embedder

        Args:
            glove_filepath (str): filepath where the glove.6B.50d file can be found
            saved_filepath (str): filepath where the custom made file can be found containing the custom randomized char-to-vec data. 
            If none exists, a new one can be made using the class method "initialize_new_char_to_vec".
            max_tokens_per_utternace (int, optional): the max number of tokens/words in one utterance . Defaults to 87 (max for VUB data).
            max_char_per_token (int, optional): the max number of characters in one token. Defaults to 21 (max for VUB data).
        """
        
        # load Glove dictionary
        glove_word_embedding = {}
        with open(glove_filepath) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = torch.tensor(np.fromstring(coefs, "f", sep=" "))
                glove_word_embedding[word] = coefs
        
        # load randomized char-to-vec data
        self.char_to_feature_vocab = {}
        with open(saved_filepath, "r") as fp:
            self.char_to_feature_vocab = {k: torch.tensor(v) for k, v in json.load(fp).items()}

        # initialize parameters
        self.max_tokens_per_utternace = max_tokens_per_utternace
        self.max_char_per_token = max_char_per_token
        self.glove_word_embedding = glove_word_embedding
        self.word_embedding_dimension_size = len(glove_word_embedding["a"])
        self.word_tokenizer = get_tokenizer("basic_english")
        self.char_to_vector_size = len(self.char_to_feature_vocab["a"])
        self.glove_vector_size = len(self.glove_word_embedding["dog"])


    @classmethod
    def initialize_new_char_to_vec(cls,feature_size_per_char:int,saved_filepath:str) -> None:
        """method to create a new randomized char-to-vec 
        Args:
            feature_size_per_char (int): size of the vector for one char
            saved_filepath (str): the filepath/name to store the char-to-vec in
        """
        characters_vocab = string.printable
        random_char_distribution = np.random.uniform(-0.5,0.5,[len(characters_vocab),feature_size_per_char])
        char_to_feature_vocab = {n:random_char_distribution[m] for m,n in enumerate(characters_vocab)}
        with open(saved_filepath, "w") as fp:
            json.dump({k: v.tolist() for k, v in char_to_feature_vocab.items()} , fp) 


    def embed_char(self,char:str) -> torch.Tensor:
        """ returns the char-to-vec embedding of one character. if the character is not available, a zero tensor will be returned
        Args:
            char (str): the character to embed using the char-to-vec
        Raises:
            ValueError: if a string with more than one char is used as input
        Returns:
            torch.Tensor: the char-to-vec tensor for the given character
        """
        if len(char) == 1:
            result = self.char_to_feature_vocab.get(char)
            if not isinstance(result,torch.Tensor):
                result = torch.zeros([self.char_to_vector_size ])
            return result
        else: raise ValueError("variable char contains string with more than one char")
    

    def tokenize_utterance(self,utterance:str) -> List[str]:
        """divide a utterance/string into its tokens
        Args:
            utterance (str): the string to be tokenized
        Returns:
            List[str]: the list of tokens for the given utterance/string
        """
        return self.word_tokenizer(utterance)
    

    def embed_token(self,token:str) -> torch.Tensor:
        """returns the Glove word-to-vec embedding of one token. if token is not available a zero tensor will be returned
        Args:
            token (str): token to be embedded
        Returns:
            torch.Tensor: the Glove word-to-vec tensor for the given token
        """
        result =  self.glove_word_embedding.get(token)
        if not isinstance(result,torch.Tensor):
                result = torch.zeros([self.word_embedding_dimension_size])
        return result
    

    def embed_tokens(self,tokens:List[str])-> torch.Tensor:
        """returns the Glove word-to-vec embedding of a list of tokens. if a token is not available a zero tensor 
        for that token will be returned.
        Args:
            tokens (List[str]): the list of tokens to be embedded
        Returns:
            torch.Tensor: the 2D Glove word-to-vec tensor for the given tokens, one rowtensor per token
        """
        # padding with zeros
        results = torch.zeros([self.max_tokens_per_utternace,self.glove_vector_size])
        for i in range(len(tokens)):
            results[i] = self.embed_token(tokens[i])
        return results
    

    def token_to_char_embedding(self,token:str)->torch.Tensor:
        """returns the 2D tensor char-to-vec embedding of all characters in a token
        Args:
            token (str): the token to be embedded
        Returns:
            torch.Tensor: 2D tensor with size 
                          [max_char_per_token,char_to_vector_size] and padding on left and right 
                          containing the character embedding for all characters in the token 
        """
        word_embedding = torch.zeros([self.max_char_per_token,self.char_to_vector_size])
        left_padding = (self.max_char_per_token-len(token))//2
        for i in range(len(token)):
            features = self.embed_char(token[i])
            word_embedding[left_padding+i]= features
        return word_embedding
    

    def embed_tokens_characters(self,tokens:List[str])-> torch.Tensor:
        """returns the 3D tensor char-to-vec embedding of all characters of all tokens in a list of tokens
        Args:
            tokens (List[str]): the list of tokens to be embedded
        Returns:
            torch.Tensor: 3D tensor with size [max_tokens_per_utternace,max_char_per_token,char_to_vector_size]
             and padding on left and right containing the character embedding for all characters for each token in the list. 
        """
        results = torch.zeros(self.max_tokens_per_utternace,self.max_char_per_token,self.char_to_vector_size)
        for i in range(len(tokens)):
            results[i]= self.token_to_char_embedding(tokens[i])
        return results
