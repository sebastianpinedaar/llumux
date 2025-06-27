
from datasets import load_dataset
import numpy as np

from .base_dataset import BaseDataset
from ..utils.constants import *

class ListwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 list_size: int = 3,
                 model_list: list = None):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.model_list = model_list
        self.random_sample = random_sample
        self.list_size = list_size
        self.fixed_len = 10000

        if dataset_name == "lmarena-ai/arena-human-preference-55k":
            dataset_before_split = load_dataset(dataset_name)["train"]
            self.dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)[split]
        elif dataset_name == "llm-blender/mix-instruct":
            dataset_before_split = load_dataset(dataset_name)["train"]
            self.dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)[split]  
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
                             
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model_a: the first model to compare
        - model_b: the second model to compare
        - prompt: the prompt to compare the models
        - winner_model_a: 1 if model_a is better, -1 if model_b is better
        """
        
        if self.dataset_name == "llm-blender/mix-instruct":
            idx = np.random.randint(0, len(self.dataset))
            model_idxs = [np.random.randint(0, 12) for _ in range(self.list_size)]
            models = [self.dataset[idx]["candidates"][model_idx]["model"] for model_idx in model_idxs]
            target = [self.dataset[idx]["candidates"][model_idx]["scores"]["bertscore"] for model_idx in model_idxs]
            
            # the target is the ground truth order
            item = self.dataset[idx]
            item = { 
                "prompt": item["instruction"] + ". "+ item["input"], \
                "target": target, \
                "models": models
            } 
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item


if __name__ == "__main__":

    dataset = ListwiseDataset("llm-blender/mix-instruct", split="train", test_size=0.1, seed=1)
    print(next(iter(dataset)))
    print(dataset.get_number_of_models())