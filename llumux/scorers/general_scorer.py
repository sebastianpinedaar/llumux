import torch
import torch.nn as nn
import numpy as np

from typing import List
from sklearn.preprocessing import OneHotEncoder

from ..losses import loss_functions_map
from ..utils import LAST_HIDDEN_DIM
from .base_scorer import BaseScorer

class GeneralScorer(BaseScorer):
    """
    General scorer class for evaluating models based on prompts.
    It supports different embedding strategies and loss functions.
    If list_size == 1, it behaves like a pointwise scorer, and supports pointwise losses.
    If list_size == 2, it behaves like a pairwise scorer, and supports pairwise losses.
    If list_size > 2, it behaves like a listwise scorer, and supports listwise losses.
    It does not support pairwise datasets input.
    """
    
    def __init__(self, model_list: List[str] = None,
                    use_frozen_embedder: bool = False,
                    hidden_size: int = 32, 
                    output_size: int = 1,
                    max_length: int = 512,
                    prompt_embedder_name: str ="bert-base-uncased",
                    loss_fun_name: str ="list_mle",
                    embeddings_merge_strategy: str ="concat",
                    device: str ="cuda"):
        super(GeneralScorer, self).__init__(use_frozen_embedder=use_frozen_embedder)

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

    def forward(self, prompts: List[str], 
                models: List[str], 
                targets: List[List[float]]=None,
                **kwargs):
        prompt_embedding = self.get_prompt_embedding(prompts)
        prompt_embedding = self.ln1(prompt_embedding)
        prompt_embedding = self.fc1_prompt(prompt_embedding)
        scores = []
        for model in models:
            score = self.score(prompt_embedding, model)
            scores.append(score)
        scores = torch.stack(scores, dim=1).to(self.device)[...,0].float()
        
        if targets is None:
            loss = None
        else:
            targets = torch.stack(targets, dim=1).to(self.device).float()
            loss = self.loss_fn(scores, targets)

        return scores, loss

    def score(self, prompt_embeddings: torch.Tensor, model_names: List[str]):
        model_encodings = self.model_encoder.transform(np.array(model_names).reshape(-1, 1)).toarray()
        model_encodings = torch.tensor(model_encodings).to(self.device).float()
        model_embeddings = self.fc1_model(model_encodings)
        
        if self.embeddings_merge_strategy == "multiply":
            x = torch.multiply(prompt_embeddings, model_embeddings)
        elif self.embeddings_merge_strategy == "concat":
            x = torch.cat([prompt_embeddings, model_embeddings], dim=1)
        elif self.embeddings_merge_strategy == "add":
            x = prompt_embeddings + model_embeddings
        else:
            raise ValueError(f"Unknown embeddings merge strategy: {self.embeddings_merge_strategy}")

        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)
        return out

    def get_config(self):
        return {
            "model_list": self.model_list,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "prompt_embedder_name": self.prompt_embedder_name,
            "loss_fun_name": self.loss_fun_name,
            "device": self.device,
            "use_frozen_embedder": self.use_frozen_embedder,
            "max_length": self.max_length,
            "embeddings_merge_strategy": self.embeddings_merge_strategy
        }
