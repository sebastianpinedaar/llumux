
import numpy as np
from pathlib import Path

from .base_dataset import BaseDataset

class ListwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str,
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 list_size: int = 3,
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
        self.list_size = list_size
        self.fixed_len_train = fixed_len_train
        self.fixed_len_eval = fixed_len_eval
        self.score_name = score_name
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.model_hub_name = model_hub_name
        self.target_scale = target_scale
        self.dataset = self.get_dataset(dataset_name, split, test_size, seed)

                         
    def __getitem__(self, idx):
        """Returns a dictionary containing the following keys:
        - prompts: the prompt to compare the models 
        - targets: the scores of the models
        - models: the models to compare
        """
        
        if self.dataset_name == "llm-blender/mix-instruct":
            idx = np.random.randint(0, len(self.dataset))
            item = self.dataset[idx]
            model_idxs = [np.random.randint(0, self.num_models) for _ in range(self.list_size)]
            models = [self.dataset[idx]["candidates"][model_idx]["model"] for model_idx in model_idxs]
            
            if self.score_name.endswith("complexity"):
                target = [
                    self.get_text_complexity(item["candidates"][model_idx]["text"]) \
                        for model_idx in model_idxs
                        ]
            else:
                target = [item["candidates"][model_idx]["scores"][self.score_name] \
                          for model_idx in model_idxs]
            
            # the target is the ground truth order
            item = { 
                "prompts": item["instruction"] + ". "+ item["input"], \
                "targets": target, \
                "models": models
            }
        elif self.dataset_name == "custom":
            item = self.dataset[idx]
            available_models = list(item["candidates"].keys())
            models = np.random.choice(available_models, self.list_size).tolist()

            if self.score_name.endswith("complexity"):
                target = [
                    self.get_text_complexity(item["candidates"][model]["text"]) \
                    for model in models
                ]
            else:
                target = [item["candidates"][model]["scores"][self.score_name] \
                           for model in models]
            
            item = { 
                "prompts": item["prompt"],
                "targets": target,
                "models": models
            }            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item