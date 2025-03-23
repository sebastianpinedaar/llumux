
from datasets import load_dataset
import torch
import numpy as np

from .base_dataset import BaseDataset

NUMBER_OF_MODELS = {
    "lmarena-ai/arena-human-preference-55k": 64
}

class PairwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.model_count = None

        if dataset_name == "lmarena-ai/arena-human-preference-55k":
            dataset_before_split = load_dataset(dataset_name)["train"]
            self.dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)[split]
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
                             
    def get_number_of_models(self):
        return NUMBER_OF_MODELS[self.dataset_name]
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model_a: the first model to compare
        - model_b: the second model to compare
        - prompt: the prompt to compare the models
        - winner_model_a: 1 if model_a is better, -1 if model_b is better
        """

        if self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            item = self.dataset[idx]
            item = { 
                "prompt": item["prompt"], \
                "target": item["winner_model_a"]-1*item["winner_model_b"], \
                "model_a": item["model_a"],
                "model_b": item["model_b"]
            }
            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item

    def collect_models(self):
        if self.model_count is None:
            models = []
            for item in self.dataset:
                models.append(item["model_a"])
                models.append(item["model_b"])
            
            self.model_list = np.unique(models).tolist()

        return self.model_list


if __name__ == "__main__":

    dataset = PairwiseDataset("lmarena-ai/arena-human-preference-55k", split="train", test_size=0.1, seed=1)
    print(dataset.get_number_of_models())