import numpy as np

from .base_dataset import BaseDataset

class PointwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False):
        super().__init__(dataset_name, split, test_size, seed, random_sample)
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model: the model to compare
        - prompt: the prompt to compare the models
        - score: the score of the model
        """
        if self.random_sample:
            idx = np.random.randint(0, len(self.dataset))
        
        if self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            raise NotImplementedError("PointwiseDataset is not implemented for lmarena-ai/arena-human-preference-55k dataset")
        
        elif self.dataset_name == "llm-blender/mix-instruct":
            model_id = np.random.randint(0, NUMBER_OF_MODELS[self.dataset_name])
            item = self.dataset[idx]
            item = { 
                "prompt": item["instruction"] + ". "+ item["input"], \
                "target": item["candidates"][model_id]["scores"]["bertscore"],
                "model": item["candidates"][model_id]["model"]
            } 
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item