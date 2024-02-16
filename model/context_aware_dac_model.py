#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains the deep learning model based on the original paper: 
# "Dialogue Act Classification with Context-Aware Self-Attention" by Raheja & Tetreault, NAACL 2019
#######################################################################################################

from typing import Dict, List, Union
import os
import torch
import torch.nn as nn

from .glove_and_char_embedder import Glove_and_Char_Embedder

class ContextAwareDAC(nn.Module):

    def __init__(self,labels:Dict[str,int],hidden_size=768,max_tokens_per_utternace=87,device=torch.device("cpu")):
        """ initialises the model

        Args:
            labels (List[str]): the list of uttterance intent labels/classes
            hidden_size (int, optional): hyperparameter to set the size of hidden layers. Defaults to 768. unused in current implementation.
            max_tokens_per_utternace (int, optional): max token/word length of an utterance. Defaults to 87.
            device (torch.device, optional): Pytorch device object referring to the processing unit to use (e.g. CUDA, mps, cpu). 
            Defaults to torch.device("cpu").
        """
        
        super(ContextAwareDAC, self).__init__()
        
        self.name = "CASA-dialogue-ACT-classifier"
        self.version = "V3.0"
        self.num_classes=len(labels)
        self.labels:Dict[str,int] = labels 
        self.device = device
        self.max_tokens_per_utternace = max_tokens_per_utternace
        
        # optimizer has to be set manually prior to training.
        # including the optimizer in the model allows for saving the model as a pickle file with the current optimizer state included
        self.optimizer = None

        # load the embedder object used for word-to-vec and char-to-vec
        self.embedder = Glove_and_Char_Embedder(os.path.join("model","glove","glove.6B.50d.txt"),os.path.join("model","char_embedding.json"))
        
        # CNN used to reduce dimensionality of Char-to-vec in utterance level part of paper
        self.utt_cnn1 = nn.Conv2d(in_channels=max_tokens_per_utternace, out_channels=max_tokens_per_utternace,kernel_size=(5, 5))
        self.utt_cnn2 = nn.ReLU()
        self.utt_cnn3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.utt_cnn4 = nn.Conv2d(in_channels=max_tokens_per_utternace, out_channels=max_tokens_per_utternace,kernel_size=(5, 5))
        self.utt_cnn5 = nn.ReLU()
        self.utt_cnn6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # GRU layer used as last step in utterance level part as described in paper
        self.utt_rnn = nn.GRU(
            input_size=58, 
            hidden_size=128, 
            num_layers=1, 
            bidirectional=True,
            batch_first=True
        )

        # multi layer perceptron layers used for equation 5 in context aware attention part of original paper
        self.context_nn1 = nn.Linear(in_features=256, out_features=128, bias=False)
        self.context_nn2 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.context_nn3= nn.Linear(in_features=128, out_features=128, bias=False)
        
        # 2D representation linear projection to a 1D embedding (denoted as hi) in context aware attention part, as described in paper
        self.linear_projection = nn.Linear(in_features=256, out_features=1, bias=True)

        # conversation-level context initial states (gi in paper) 
        self.gx = torch.randn((2, 2, 128), device=self.device, requires_grad=True)

        # final GRU layer as used in Conversation-level RNN of original paper
        self.conversation_rnn = nn.GRU(
            input_size=1,
            hidden_size=128, 
            num_layers=1, 
            bidirectional=True,
            batch_first=True
        )

        # Conditional Random Fields(CRF) in original paper are replaced by standard deep learning classifier 
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=256*2, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=self.num_classes)
        ])


    def forward(self, utterance:Union[str,List[str]])-> torch.Tensor:
        """ forward pass to predict the utterance intent labels/classes
        Args:
            utterance (Union[str,List[str]]): a single utterance of list of utterances in one dialogue
        Raises:
            Exception: raised if input is not a string or list of strings
        Returns:
            torch.Tensor: the predicted likelihood distribution for the given utterance(s) labels/classes
        """

        # --------------------------------------------------------------------------------------------
        # Utterance-level RNN
        # --------------------------------------------------------------------------------------------
        
        # create embedding of utterances and its characters
        token_embeddings = None
        dim = None
        if isinstance(utterance,str):
            dim = 1
            tokens = self.embedder.tokenize_utterance(utterance)
            token_embeddings = self.embedder.embed_tokens(tokens) 
            token_char_embeddings = self.embedder.embed_tokens_characters(tokens)
            
        elif isinstance(utterance,list) or isinstance(utterance,tuple):
            dim = 2
            tokens = []
            token_embeddings =  torch.zeros([len(utterance),self.max_tokens_per_utternace,50]) # 50 is features per token used by glove
            token_char_embeddings = torch.zeros([len(utterance),self.max_tokens_per_utternace, 21, 50]) #W 21 is max char length, 50 is features per char embedding
            for i in range(len(utterance)):
                tokens = self.embedder.tokenize_utterance(utterance[i])
                token_embeddings[i] = self.embedder.embed_tokens(tokens)
                token_char_embeddings[i] = self.embedder.embed_tokens_characters(tokens)
                                            
        else:
            raise Exception("model input must be string or list of strings")
        
        token_embeddings = token_embeddings.to(self.device)
        token_char_embeddings = token_char_embeddings.to(self.device)
        
        #  reduce dimensionality of Char-to-vec 
        x = self.utt_cnn1(token_char_embeddings) 
        x = self.utt_cnn2(x)
        x = self.utt_cnn3(x)
        x = self.utt_cnn4(x)
        x = self.utt_cnn5(x)
        cnn_results = self.utt_cnn6(x) 
        
        # concatenate word-to-vec embeddings with char-to-vec embeddings as described in paper: 
        # Named entity recognition with bidirectional lstm-cnns by (Chiu and Nichols, 2016)
        RNN_inputs = torch.cat([token_embeddings,cnn_results.flatten(2)],dim=dim)
        
        # get hidden states from utterance RNN
        _,hidden_states = self.utt_rnn(RNN_inputs)
        bidirectional_hidden_states = hidden_states.transpose(1,0).flatten(1) # Formula 3 in paper (GRU already returns bidirectional Hidden units)


        # --------------------------------------------------------------------------------------------
        # Context-aware Self-attention and Conversation-level RNN
        # --------------------------------------------------------------------------------------------
        
        features = torch.empty((0,256*2), device=self.device,requires_grad=True)
        gx = self.gx # load conversation-level context 
        
        # loop over each utterance in the batch
        for i, x in enumerate(bidirectional_hidden_states): 
            
            # get sentence representation as 2d-matrix and project it linearly
            Hx = x.unsqueeze(0) 
            a = self.context_nn1(Hx)
            b = self.context_nn2(gx[0].detach().unsqueeze(1))
            S = self.context_nn3(torch.tanh(a + b)) # Formula 5 in paper
            A = S.softmax(dim=-1) # Formula 6 in paper
            Mx = torch.matmul(A.permute(0, 2, 1), Hx) # Formula 7 in paper
            
            # project 2D representation to 1D by using fully connected layer
            one_D_embedding  = self.linear_projection(Mx)  # fuly connected layer as described between formula 7 and 8

            # conversation level GRU
            _, gx = self.conversation_rnn(input=one_D_embedding, hx=gx.detach()) # formula 8,9,10 in paper
            
            # concat current utterance's last hidden state to the features vector
            features = torch.cat((features, gx.view(1, -1)), dim=0) 
        self.gx = gx # save hidden state of conversation-level RNN
        logits = self.classifier(features)
        return logits
    

    def reset_device(self,device:torch.device):
        """resets the device to use (for instance when existing model is loaded on different machine)
        Args:
            device (torch.device): the new device to use
        """
        self.device = device
    

    def set_optimizer(self,optimizer:torch.optim.Optimizer):
        """sets the optimizer to use during training
        Args:
            optimizer (torch.optim.Optimizer): Pytorch optimizer to use during training
        """
        self.optimizer=optimizer


    def get_optimizer(self) -> torch.optim.Optimizer:
        """get the optimizer object used by the model

        Returns:
            torch.optim.Optimizer: the optimizer object used by the model
        """
        if self.optimizer:
            return self.optimizer 
        else:
            raise Exception("the optimizer in the ContextAwareDAC object was not set")