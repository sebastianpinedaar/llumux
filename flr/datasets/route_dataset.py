import numpy as np

from .base_dataset import BaseDataset
from ..utils.constants import *

class RouteDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 model_list: list = None):
        super().__init__(dataset_name, split, test_size, seed, random_sample, model_list)
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model: the model to compare
        - prompt: the prompt to compare the models
        - score: the score of the model
        """

        if self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            raise NotImplementedError("PointwiseDataset is not implemented for lmarena-ai/arena-human-preference-55k dataset")
        
        elif self.dataset_name == "llm-blender/mix-instruct":
            item = self.dataset[idx]
            item["candidates"] = [item["candidates"]]
            item.update({
                "prompt": item["instruction"] + ". "+ item["input"]
                }
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item