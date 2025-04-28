import numpy as np
from datasets import load_dataset

NUMBER_OF_MODELS = {
    "lmarena-ai/arena-human-preference-55k": 64,
    "llm-blender/mix-instruct": 12
}


class BaseDataset:
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
        raise NotImplementedError("This method should be implemented in the subclass")
    

    def collect_models(self):
        #TODO: this should be done in the __init__ method
        #TODO. improve collection
        if self.model_list is None:
            models = []
            for i in range(len(self)):
                item = self[i]

                if "model_a" in item.keys():
                    models.append(item["model_a"])
                if "model_b" in item.keys():
                    models.append(item["model_b"])
                if "model" in item.keys():
                    models.append(item["model"])
            
            self.model_list = np.unique(models).tolist()

        return self.model_list

