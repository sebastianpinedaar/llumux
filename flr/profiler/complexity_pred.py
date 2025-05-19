import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import BertModel, BertTokenizer
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import AlbertModel

from ..scorers.pointwise.scorer import PointwiseScorer
from ..losses import LOSS_FUNCTIONS


class ComplexityPredictor(PointwiseScorer):
    def __init__(self, model_list,
                    hidden_size=32, 
                    output_size=1,
                    max_length=512,
                    prompt_embedder_name="bert-base-uncased",
                    loss_fun="pairwise_logistic_loss",
                    device="cuda"):
        super(ComplexityPredictor, self).__init__(
                        model_list=model_list,
                        hidden_size=hidden_size, 
                        output_size=output_size,
                        max_length=max_length,
                        prompt_embedder_name=prompt_embedder_name,
                        loss_fun=loss_fun,
                        device=device)
        
    
    def forward(self, prompt, target, model, **kwargs):     
        len_pred = super(ComplexityPredictor, self).score(prompt, target, model, **kwargs)