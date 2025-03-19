import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from transformers import BertModel, BertTokenizer

from ...losses import pairwise_logistic_loss

LAST_HIDDEN_DIM = {
    'bert-base-uncased': 768,
    'bert-base-cased': 768
}

class BertScorer(nn.Module):
    def __init__(self, unique_models,
                    hidden_size=32, 
                    output_size=1,
                    prompt_embedder_name="bert-base-uncased",
                    loss_fn=pairwise_logistic_loss,
                    device="cuda"):
        super(BertScorer, self).__init__()

        self.unique_models = unique_models
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.prompt_embedder = BertModel.from_pretrained(prompt_embedder_name, 
                                                         torch_dtype=torch.float32, 
                                                         attn_implementation="sdpa").to(device)
        self.prompt_tokenizer = BertTokenizer.from_pretrained(prompt_embedder_name)
        self.last_hidden_state_dim = LAST_HIDDEN_DIM[prompt_embedder_name]

        self.model_encoder = OneHotEncoder()
        self.model_encoder.fit(np.array(self.unique_models).reshape(-1, 1))
        self.fc1_prompt = nn.Linear(self.last_hidden_state_dim, hidden_size).to(device)
        self.fc1_model = nn.Linear(len(self.unique_models), hidden_size).to(device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2*hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(device)
        self.ln1 = nn.LayerNorm(self.last_hidden_state_dim).to(device)
        self.loss_fn = loss_fn

    def forward(self, prompt, model_names_a, model_names_b, target,
                **kwargs):
        tokens = self.prompt_tokenizer(prompt, return_tensors='pt', padding="max_length", max_length=512, truncation=True).to(self.device)
        input_ids, token_type_ids, attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
        prompt_embedding = self.prompt_embedder(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
        prompt_embedding = prompt_embedding.last_hidden_state[:, 0, :]
        prompt_embedding = self.fc1_prompt(self.ln1(prompt_embedding))
        score_a = self.score(prompt_embedding, model_names_a)
        score_b = self.score(prompt_embedding, model_names_b)
        loss = self.loss_fn((score_a, score_b), target)

        return loss

    def score(self, prompt_embedding, model_names):
        model_encoding = self.model_encoder.transform(np.array(model_names).reshape(-1, 1)).toarray()
        model_encoding = torch.tensor(model_encoding).to(self.device).float()
        model_embedding = self.fc1_model(model_encoding)
        
        x = torch.cat([prompt_embedding, model_embedding], dim=1)
        
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc3(out)

        return out

