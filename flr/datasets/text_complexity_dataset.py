import numpy as np

from .base_dataset import BaseDataset

class TextComplexityDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 model_list: list = None,
                 target_scale: int = 1000,
                 complexizy_type: str = "length"):
        super().__init__(dataset_name, split, test_size, seed, random_sample, model_list)
        self.target_scale = target_scale
        self.complexity_type = complexizy_type
    
    def get_text_complexity(self, text: str):
        """
        Calculate the complexity of the prompt based on the specified complexity type.
        """
        if self.complexity_type == "length":
            return len(text) / self.target_scale
        elif self.complexity_type == "word_count":
            return len(text.split()) / self.target_scale
        else:
            raise ValueError(f"Complexity type {self.complexity_type} not supported")
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model: the model to compare
        - prompt: the prompt to compare the models
        - score: the score of the model
        """
        if self.random_sample:
            idx = np.random.randint(0, len(self.dataset))
        
        if self.dataset_name == "llm-blender/mix-instruct":
            item = self.dataset[idx]
            model_id = np.random.randint(0, len(item["candidates"]))
            item = { 
                "prompt": item["instruction"] + ". "+ item["input"], \
                "target": self.get_text_complexity(item["candidates"][model_id]["text"]),
                "model": item["candidates"][model_id]["model"]
            } 

        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item