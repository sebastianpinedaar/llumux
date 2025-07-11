import numpy as np
from pathlib import Path

from .base_dataset import BaseDataset

class PointwiseDataset(BaseDataset):
    """Dataset class for pointwise comparisons."""
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 fixed_len_train: int = 10000,
                 fixed_len_eval: int = 1000,
                 score_name: str = "bertscore",
                 model_hub_name: str = None,
                 dataset_path: str = None,
                 target_scale: float = 1.,
                 **kwargs):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.random_sample = random_sample
        self.fixed_len_train = fixed_len_train
        self.fixed_len_eval = fixed_len_eval
        self.score_name = score_name
        self.dataset_path = Path(dataset_path)
        self.model_hub_name = model_hub_name
        self.target_scale = target_scale
        self.dataset = self.get_dataset(dataset_name, split, test_size, seed)

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
            model_id = np.random.randint(0, self.num_models)
            item = self.dataset[idx]
            item = { 
                "prompts": item["instruction"] + ". "+ item["input"], \
                "targets": item["candidates"][model_id]["scores"]["bertscore"],
                "models": item["candidates"][model_id]["model"]
            }

        elif self.dataset_name == "custom_flr":
            item = self.dataset[idx]
            available_models = list(item["candidates"].keys())
            model = np.random.choice(available_models).item()

            if self.score_name.endswith("complexity"):
                target = self.get_text_complexity(item["candidates"][model]["text"])
            else:
                target = item["candidates"][model]["scores"][self.score_name]
            
            item = { 
                "prompts": item["prompt"],
                "targets": target,
                "models": model
            }   
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item