
from datasets import load_dataset
import numpy as np

from .base_dataset import BaseDataset

NUMBER_OF_MODELS = {
    "lmarena-ai/arena-human-preference-55k": 64
}

class PairwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 model_list: list = None):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.model_list = model_list
        self.random_sample = random_sample
        self.fixed_len = 10000

        if dataset_name == "lmarena-ai/arena-human-preference-55k":
            dataset_before_split = load_dataset(dataset_name)["train"]
            self.dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)[split]
        elif dataset_name == "llm-blender/mix-instruct":
            dataset_before_split = load_dataset(dataset_name)["train"]
            self.dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)[split]  
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
                             
    def get_number_of_models(self):
        return NUMBER_OF_MODELS[self.dataset_name]
    
    def __len__(self):
        if self.random_sample:
            return self.fixed_len if self.split == "train" else 100
        elif self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            return len(self.dataset)
        elif self.dataset_name == "llm-blender/mix-instruct":
            return len(self.dataset)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
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
            assert self.random_sample == True, "Random sampling is required for llm-blender/mix-instruct dataset"
            model_a = np.random.randint(0, 12)
            model_b = np.random.randint(0, 12)
            item = self.dataset[idx]
            item = { 
                "prompt": item["instruction"] + ". "+ item["input"], \
                "target": -np.sign(item["candidates"][model_a]["scores"]["logprobs"]-item["candidates"][model_b]["scores"]["logprobs"]).item(), \
                "model_a": item["candidates"][model_a]["model"],
                "model_b": item["candidates"][model_b]["model"]
            } 
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item

    def collect_models(self):
        #TODO: this should be done in the __init__ method
        #TODO. improve collection
        if self.model_list is None:
            models = []
            for i in range(len(self)):
                item = self[i]
                models.append(item["model_a"])
                models.append(item["model_b"])
            
            self.model_list = np.unique(models).tolist()

        return self.model_list


if __name__ == "__main__":

    dataset = PairwiseDataset("lmarena-ai/arena-human-preference-55k", split="train", test_size=0.1, seed=1)
    print(dataset.get_number_of_models())