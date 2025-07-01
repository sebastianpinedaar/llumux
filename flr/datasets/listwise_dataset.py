
import numpy as np

from .base_dataset import BaseDataset

class ListwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 list_size: int = 3,
                 fixed_len_train: int = 10000,
                 fixed_len_eval: int = 1000):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.random_sample = random_sample
        self.list_size = list_size
        self.fixed_len_train = fixed_len_train
        self.fixed_len_eval = fixed_len_eval
        self.dataset = self.get_dataset(dataset_name, split, test_size, seed)

                             
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