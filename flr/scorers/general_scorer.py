import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ..losses import loss_functions_map
from .base_scorer import BaseScorer
from ..utils import LAST_HIDDEN_DIM

class GeneralScorer(BaseScorer):
    def __init__(self, model_list,
                    hidden_size=32, 
                    output_size=1,
                    max_length=512,
                    prompt_embedder_name="bert-base-uncased",
                    loss_fun_name="list_mle",
                    embeddings_merge_strategy="concat",
                    device="cuda"):
        super(GeneralScorer, self).__init__()

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

        if embeddings_merge_strategy == "concat":
            self.fc2 = nn.Linear(2*hidden_size, hidden_size).to(device)
        else:
            self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(device)
        self.ln1 = nn.LayerNorm(self.last_hidden_state_dim).to(device)
        self.loss_fun_name = loss_fun_name
        self.embeddings_merge_strategy = embeddings_merge_strategy
        self.loss_fn = loss_functions_map[loss_fun_name]()
        self.initialize_prompt_embedder()
        self.to(device)

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

    def forward(self, prompt, models, target=None,
                **kwargs):
        prompt_embedding = self.get_prompt_embedding(prompt)
        prompt_embedding = self.ln1(prompt_embedding)
        prompt_embedding = self.fc1_prompt(prompt_embedding)
        scores = []
        for model in models:
            score = self.score(prompt_embedding, model)
            scores.append(score)
        scores = torch.stack(scores, dim=1).to(self.device)[...,0].float()
        target = torch.stack(target, dim=1).to(self.device).float()
        
        if target is None:
            loss = None
        else:
            loss = self.loss_fn(scores, target)

        return scores, loss

    def score(self, prompt_embedding, model_names):
        model_encoding = self.model_encoder.transform(np.array(model_names).reshape(-1, 1)).toarray()
        model_encoding = torch.tensor(model_encoding).to(self.device).float()
        model_embedding = self.fc1_model(model_encoding)
        
        if self.embeddings_merge_strategy == "multiply":
            x = torch.multiply(prompt_embedding, model_embedding)
        elif self.embeddings_merge_strategy == "concat":
            x = torch.cat([prompt_embedding, model_embedding], dim=1)
        elif self.embeddings_merge_strategy == "add":
            x = prompt_embedding + model_embedding
        else:
            raise ValueError(f"Unknown embeddings merge strategy: {self.embeddings_merge_strategy}")

        hidden_out = self.fc2(x)
        out = self.relu(hidden_out)
        out = self.fc3(out) + hidden_out
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
