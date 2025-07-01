
from datasets import load_dataset
import numpy as np

from .base_dataset import BaseDataset

class PairwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.random_sample = random_sample
        self.fixed_len = 10000
        self.dataset = self.get_dataset(dataset_name, split, test_size, seed)
                             
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model_a: the first model to compare
        - model_b: the second model to compare
        - prompt: the prompt to compare the models
        - winner_model_a: 1 if model_a is better, -1 if model_b is better
        """
        if self.random_sample:
            idx = np.random.randint(0, len(self.dataset))
        
        if self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            item = self.dataset[idx]
            item = { 
                "prompt": item["prompt"], \
                "target": item["winner_model_a"]-1*item["winner_model_b"], \
                "model_a": item["model_a"],
                "model_b": item["model_b"]
            }
        elif self.dataset_name == "llm-blender/mix-instruct":
            model_a = np.random.randint(0, 12)
            model_b = np.random.randint(0, 12)
            item = self.dataset[idx]
            item = { 
                "prompt": item["instruction"] + ". "+ item["input"], \
                "target": -np.sign(item["candidates"][model_a]["scores"]["bertscore"]-item["candidates"][model_b]["scores"]["bertscore"]).item(), \
                "model_a": item["candidates"][model_a]["model"],
                "model_b": item["candidates"][model_b]["model"]
            } 
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item
