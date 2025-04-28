import pandas as pd
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt

from flr.datasets.pairwise_dataset import PairwiseDataset

class BinaryMatrix(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.num_items = num_items
        self.num_users = num_users
        self.temp = 10
        self.num_factors = 100
        self.binary_selector = nn.Parameter(torch.randn(num_users,self.num_factors))
    def forward(self, x, idx):
        w = nn.Softmax(dim=0)(self.temp*self.binary_selector[idx])
        a = x.T @ w
        #loss =(a@a.T).mean()
        loss =torch.corrcoef(a).mean()
        return a, loss
    

def build_matrix(load_from_file=None):
    if load_from_file is not None:
        x = np.load(load_from_file)
        return x

    dataset_name = "llm-blender/mix-instruct"    
    dataset_size = 100000
    dataset= load_dataset(dataset_name)["train"]
    observations = []
    score_names = ['logprobs']#, 'rougeL', 'rouge2', 'rougeLsum', 'rouge1', 'bleu', 'bertscore', 'bleurt', 'bartscore']
    model_names = ['vicuna-13b-1.1', 'flan-t5-xxl', 'stablelm-tuned-alpha-7b', 'koala-7B-HF', 'dolly-v2-12b', 'chatglm-6b', \
                    'oasst-sft-4-pythia-12b-epoch-3.5', 'llama-7b-hf-baize-lora-bf16', 'moss-moon-003-sft', 'mpt-7b', 'mpt-7b-instruct', 'alpaca-native']
    models_scores = defaultdict(lambda:defaultdict(list))

    for i, x in enumerate(dataset):
        if i<dataset_size:
            for score in score_names:
                for c in x["candidates"]:
                    models_scores[score][c["model"]].append(c["scores"][score])

    scores_matrices = defaultdict(list)
    for score in score_names:
        for model in model_names:
                scores_matrices[score].append(models_scores[score][model])

  
    x = scores_matrices["logprobs"]
    x = np.array(x).T
    #save array
    np.save("logprobs.npy", x)
    return x

if __name__ == "__main__":

    x = build_matrix("logprobs.npy")

    num_users, num_items = x.shape
    num_factors = 4
    lr = 0.0001
    epochs = 50000
    device = "cuda"

    model = BinaryMatrix(num_users, num_items)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model = model.to(device)
    x = torch.tensor(x, dtype=torch.float32)
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        idx = np.random.choice(x.shape[0], 10000, replace=False)
        x_batch = x[idx]
        x_batch = x_batch.to(device)
        idx = torch.tensor(idx, dtype=torch.long).to(device)
        prediction, loss = model(x_batch, idx)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
