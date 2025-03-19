
from datasets import load_dataset
import torch
import numpy as np

from .base_dataset import BaseDataset

NUMBER_OF_MODELS = {
    "lmarena-ai/arena-human-preference-55k": 64
}

class PairwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, split: str, test_size: float, seed: int):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.model_count = None
        self.dataset = load_dataset(dataset_name, split=split)
        self.train_val_split = self.dataset.train_test_split(test_size=test_size, seed=seed)
    
    def get_number_of_models(self):
        return NUMBER_OF_MODELS[self.dataset_name]
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model_a: the first model to compare
        - model_b: the second model to compare
        - prompt: the prompt to compare the models
        - winner_model_a: 1 if model_a is better, -1 if model_b is better
        """

        if self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            item = self.train_val_split[idx]
            item = item["model_a"], item["model_b"], item["prompt"], \
                item["winner_model_a"]-1*item["winner_model_b"]
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item[idx]

    def collect_and_count_models(self):
        if self.model_count is None:
            models = []
            for item in next(iter(self.train_val_split)):
                models.extend(item["model_a"])
                models.extend(item["model_b"])
            
            self.model_list = np.unique(models).tolist()
            self.model_count = len(self.model_list)

        return self.model_list, self.model_count

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



if __name__ == "__main__":

    dataset = PairwiseDataset("lmarena-ai/arena-human-preference-55k", split="train", test_size=0.1, seed=1)
    print(dataset.get_number_of_models())