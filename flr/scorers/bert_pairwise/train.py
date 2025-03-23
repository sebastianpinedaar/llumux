import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import OneHotEncoder

from dataset.load_data import get_dataloader_from_hf, collect_models, process_inputs
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def pairwise_logistic_loss(scores_i, scores_j, y_ij):
    """
    Computes the pairwise logistic loss.

    Args:
        scores_i (torch.Tensor): Scores assigned by the model to item i.
        scores_j (torch.Tensor): Scores assigned by the model to item j.
        y_ij (torch.Tensor): Pairwise labels (1 if i should be ranked higher than j, -1 otherwise).
    
    Returns:
        torch.Tensor: The computed logistic loss.
    """
    margin = scores_i - scores_j  # Difference in scores
    loss = torch.log(1 + torch.exp(-(y_ij+0.01) * margin.reshape(-1)))  # Logistic loss
    return loss.mean()  #

# Define a simple MLP model
class LLMScorer(nn.Module):
    def __init__(self, unique_models,
                    hidden_size=32, 
                    output_size=1,
                    prompt_embedder_name="bert-base-uncased",
                    device="cuda"):
        super(LLMScorer, self).__init__()

        self.unique_models = unique_models
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.prompt_embedder = BertModel.from_pretrained(prompt_embedder_name, 
                                                         torch_dtype=torch.float32, 
                                                         attn_implementation="sdpa").to(device)
        self.prompt_tokenizer = BertTokenizer.from_pretrained(prompt_embedder_name)
        self.last_hidden_state_dim = last_hidden_state_dims[prompt_embedder_name]

        self.model_encoder = OneHotEncoder()
        self.model_encoder.fit(np.array(self.unique_models).reshape(-1, 1))
        self.fc1_prompt = nn.Linear(self.last_hidden_state_dim, hidden_size).to(device)
        self.fc1_model = nn.Linear(len(self.unique_models), hidden_size).to(device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2*hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(device)
        #batchnorm
        self.ln1 = nn.LayerNorm(self.last_hidden_state_dim).to(device)

    def forward(self, prompt, model_names_a, model_names_b,
                **kwargs):
        #with torch.autocast(device_type=self.device):
        tokens = self.prompt_tokenizer(prompt, return_tensors='pt', padding="max_length", max_length=512, truncation=True).to(self.device)
        input_ids, token_type_ids, attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
        prompt_embedding = self.prompt_embedder(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
        prompt_embedding = prompt_embedding.last_hidden_state[:, 0, :]
        prompt_embedding = self.fc1_prompt(self.ln1(prompt_embedding))
        score_a = self.score(prompt_embedding, model_names_a)
        score_b = self.score(prompt_embedding, model_names_b)
        return score_a, score_b

    def score(self, prompt_embedding, model_names):
        model_encoding = self.model_encoder.transform(np.array(model_names).reshape(-1, 1)).toarray()
        model_encoding = torch.tensor(model_encoding).to(self.device).float()
        model_embedding = self.fc1_model(model_encoding)
        
        x = torch.cat([prompt_embedding, model_embedding], dim=1)
        
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc3(out)

        return out

num_samples = 100
input_size = 10
batch_size = 4

# Initialize model, loss function, and optimizer
dataloader = get_dataloader_from_hf(batch_size=batch_size,
                                    test_size=0.9995)
unique_models = collect_models(dataloader)

model = LLMScorer(unique_models=unique_models, hidden_size=8)
criterion = nn.MSELoss()
optimizer = optim.Adafactor(model.parameters(), lr=0.01)

model.prompt_embedder.requires_grad = False
# Training loop
counter = 0
num_epochs = 20
for epoch in range(num_epochs):
    for inputs in dataloader:
        model_a, model_b, prompt, targets = process_inputs(inputs, device=model.device)
        score_a, score_b = model(prompt, model_a, model_b)
        #pairwise ranking loss
        
        loss = pairwise_logistic_loss(score_a, score_b, targets.to(model.device))
        
        print(f"Loss: {loss}")	
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        counter = 0

print("Training complete.")