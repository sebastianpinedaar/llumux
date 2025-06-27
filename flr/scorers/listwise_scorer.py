import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import BertModel, BertTokenizer
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import AlbertModel

from ..losses import PairwiseLogisticLoss
from ..losses import LOSS_FUNCTIONS
from .base_scorer import BaseScorer, LAST_HIDDEN_DIM

class ListwiseScorer(BaseScorer):
    def __init__(self, model_list,
                    hidden_size=32, 
                    output_size=1,
                    max_length=512,
                    prompt_embedder_name="bert-base-uncased",
                    loss_fun_name="list_mle",
                    device="cuda"):
        super(ListwiseScorer, self).__init__()

        self.model_list = model_list
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.prompt_embedder_name = prompt_embedder_name
        self.max_length = max_length
        self.last_hidden_state_dim = LAST_HIDDEN_DIM[prompt_embedder_name]
        self.model_encoder = OneHotEncoder(handle_unknown="ignore")
        self.model_encoder.fit(np.array(self.model_list).reshape(-1, 1))
        self.fc1_prompt = nn.Linear(self.last_hidden_state_dim, hidden_size).to(device)
        self.fc1_model = nn.Linear(len(self.model_list), hidden_size).to(device)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(2*hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(device)
        self.ln1 = nn.LayerNorm(self.last_hidden_state_dim).to(device)
        self.loss_fun_name = loss_fun_name
        self.loss_fn = LOSS_FUNCTIONS[loss_fun_name]()
        self.initialize_prompt_embedder()
        self.to(device)

    def freeze_backbone(self):
        for param in self.prompt_embedder.parameters():
            param.requires_grad = False

    def get_prompt_embedding(self, prompt):
        if self.prompt_embedder_name in ["bert-base-uncased", "albert-base-v2"]:
            tokens = self.prompt_tokenizer(prompt, return_tensors='pt', padding="max_length", max_length=self.max_length, truncation=True).to(self.device)
            input_ids, token_type_ids, attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
            prompt_embedding = self.prompt_embedder(input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)
            prompt_embedding = prompt_embedding.last_hidden_state[:, 0, :]
        elif self.prompt_embedder_name == "identity":
            prompt_embedding = torch.vstack(prompt).T.to(self.device).float()
        return prompt_embedding

    def forward(self, prompt, target, models,
                **kwargs):
        prompt_embedding = self.get_prompt_embedding(prompt)
        prompt_embedding = self.ln1(prompt_embedding)
        prompt_embedding = self.fc1_prompt(prompt_embedding)
        scores = []
        for model in models:
            score = self.score(prompt_embedding, model)
            scores.append(score)
        scores = torch.stack(scores, dim=1).to(self.device)[...,0]
        target = torch.stack(target, dim=1).to(self.device)
        loss = self.loss_fn(scores, target)

        return scores, loss

    def score(self, prompt_embedding, model_names):
        model_encoding = self.model_encoder.transform(np.array(model_names).reshape(-1, 1)).toarray()
        model_encoding = torch.tensor(model_encoding).to(self.device).float()
        model_embedding = self.fc1_model(model_encoding)
        
        x = torch.cat([prompt_embedding, model_embedding], dim=1)
        
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def get_config(self):
        return {
            "model_list": self.model_list,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "prompt_embedder_name": self.prompt_embedder_name,
            "loss_fun_name": self.loss_fun_name,
            "device": self.device
        }
