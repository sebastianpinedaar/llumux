from pathlib import Path
import numpy as np

from .base_dataset import BaseDataset

class PairwiseDataset(BaseDataset):
    """Dataset class for pairwise comparisons."""
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
                 **kwargs
                 ):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.random_sample = random_sample
        self.fixed_len_train = fixed_len_train
        self.fixed_len_eval = fixed_len_eval
        self.score_name = score_name
        self.model_hub_name = model_hub_name
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.target_scale = target_scale
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
                "prompts": item["prompt"], \
                "targets": item["winner_model_a"]-1*item["winner_model_b"], \
                "models_a": item["model_a"],
                "models_b": item["model_b"]
            }
        elif self.dataset_name == "llm-blender/mix-instruct":
            model_a = np.random.randint(0, self.num_models)
            model_b = np.random.randint(0, self.num_models)
            item = self.dataset[idx]
            item = { 
                "prompts": item["instruction"] + ". "+ item["input"], \
                "targets": np.sign(item["candidates"][model_b]["scores"][self.score_name]\
                                    -item["candidates"][model_a]["scores"][self.score_name]).item(), \
                "models_a": item["candidates"][model_a]["model"],
                "models_b": item["candidates"][model_b]["model"]
            }
        elif self.dataset_name == "custom_flr":
            item = self.dataset[idx]
            available_models = list(item["candidates"].keys())
            models = np.random.choice(available_models, 2).tolist()

            if self.score_name.endswith("complexity"):
                targets = [
                    self.get_text_complexity(item["candidates"][model]["text"]) \
                    for model in models
                ]
            else:
                targets = [item["candidates"][model]["scores"][self.score_name] \
                           for model in models]
            
            item = { 
                "prompts": item["prompt"],
                "targets": np.sign(targets[1]-targets[0]),
                "models_a": models[0],
                "models_b": models[1]
            }   
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item
