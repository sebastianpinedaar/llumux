import pandas as pd
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt

from flr.datasets.pairwise_dataset import PairwiseDataset
from kmedoids import kmedoids

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        
        self.user_factors = nn.Parameter(torch.randn(num_users, num_factors))
        self.item_factors = nn.Parameter(torch.randn(num_items, num_factors))
    def forward(self, x):
        a = self.user_factors @ self.item_factors.t()
        loss = loss_fn(a, x)
        return a, loss


if __name__ == "__main__":

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
    del scores_matrices
    del models_scores

    #train sparse amtrix factorization model



        # Hyperparameters
    num_users, num_items = x.shape
    num_factors = 4
    lr = 0.001
    epochs = 50000

    # Mask for observed entries
    mask = x != 0

    # Model and optimizer
    model = MatrixFactorization(num_users, num_items, num_factors)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()
    x = torch.tensor(x, dtype=torch.float32)
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction, loss = model(x)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    #kmedoids clustering
    w_all = model.user_factors.detach().numpy()
    idx = np.arange(len(w_all))
    idx = np.random.choice(idx, size=10000, replace=False)
    w = w_all[idx]
    c, l, ix = kmedoids(w, 20)
    plt.scatter(w[:, 0], w[:, 1], c=c, s=50, cmap='viridis')
    plt.scatter(w[ix, 0], w[ix, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title("K-Medoids Clustering")
    plt.savefig("kmedoids.png")
    print("Cluster labels:\n", l)


    for i in ix:
        pos = idx[i]
        sample = dataset[pos]
        print(f"Sample {i}:")