
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np

    
#create an uniform dataset loader
class Dataset:
    def __init__(self, dataset_name: str, split: str, test_size: float, seed: int):
        self.dataset = load_dataset(dataset_name, split=split)
        self.train_val_split = self.dataset.train_test_split(test_size=test_size, seed=seed)
        self.dataloader = torch.utils.data.DataLoader(self.train_val_split[split], batch_size=32)
        
    def get_dataloader(self):
        return self.dataloader

def get_dataloader_from_hf(
                    dataset_name: str = "lmarena-ai/arena-human-preference-55k",
                    model_name: str = "bert-base-cased",
                    test_size: float = 0.1,
                    seed: int = 1,
                    batch_size: float = 32,
                    split: str = "train"):
    
    dataset = load_dataset(dataset_name, split=split)
    train_val_split = dataset.train_test_split(test_size=test_size, seed=seed)
    dataloader = torch.utils.data.DataLoader(train_val_split[split], batch_size=batch_size)
    
    return dataloader

def collect_models(dataloader):
    models = []
    for batch in dataloader:
        models.extend(batch["model_a"])
        models.extend(batch["model_b"])
    
    unique_models = np.unique(models).tolist()

    return unique_models

def process_inputs(batch, device):
    return batch["model_a"], batch["model_b"], batch["prompt"], \
            batch["winner_model_a"]-1*batch["winner_model_b"]

if __name__ == "__main__":

    dataloader = get_dataloader_from_hf()
    print(next(iter(dataloader)))

    print(collect_models(dataloader))