import numpy as np
from pathlib import Path
from datasets import load_dataset

from ..hub.model_hub import ModelHub

class BaseDataset:
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 fixed_len_train: int = 10000,
                 fixed_len_eval: int = 1000,
                 model_hub_name: str = None,
                 dataset_path: str = None,
                 target_scale: float = 1.):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.random_sample = random_sample
        self.fixed_len_train = fixed_len_train
        self.fixed_len_eval = fixed_len_eval
        self.model_hub_name = model_hub_name
        self.dataset_path = Path(dataset_path)
        self.target_scale = target_scale
        self.dataset = self.get_dataset(dataset_name, split, test_size, seed)

    def get_dataset(self, dataset_name: str, 
                     split: str = "train", 
                     test_size: float = 0.1,
                     seed: int = 42):
        """        Loads the dataset based on the dataset name and split.
        Args:
            dataset_name (str): Name of the dataset to load.
            split (str): The split of the dataset to load (train, validation, test).
            test_size (float): Proportion of the dataset to include in the test split.
            seed (int): Random seed for reproducibility.
        Returns:
            Dataset: The loaded dataset.
        """
        if dataset_name == "lmarena-ai/arena-human-preference-55k":
            self.num_models = 64
            if split == "train":
                dataset_before_split = load_dataset(dataset_name)["train"]
                temp_dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)["train"]
                dataset = temp_dataset.train_test_split(test_size=test_size, seed=seed)["train"]
            elif split == "validation":
                dataset_before_split = load_dataset(dataset_name)["train"]
                temp_dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)["train"]
                dataset = temp_dataset.train_test_split(test_size=test_size, seed=seed)["test"]
            elif split == "test":
                dataset_before_split = load_dataset(dataset_name)["train"]
                dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)["test"]
            else:
                raise ValueError(f"Split {split} not supported for dataset {dataset_name}")
        elif dataset_name == "llm-blender/mix-instruct":
            self.num_models = 12
            dataset = load_dataset(dataset_name)[split]
        
        elif dataset_name == "custom_flr":
            assert self.model_hub_name is not None, "Custom datasets need to provide a model hub name."
            assert self.dataset_path is not None, "Custom ndatasets need to have a specified path."
            self.model_hub = ModelHub(model_hub_name=self.model_hub_name)
            self.models = self.model_hub.get_models()
            self.num_models = len(self.models)

            dataset = load_dataset("json", data_files={split: str(self.dataset_path / (split + ".json"))}, field="data")[split]
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        return dataset
    
    def __len__(self):
        if self.random_sample:
            return self.fixed_len_train if self.split == "train" else self.fixed_len_eval
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def get_text_complexity(self, text: str):
        """
        Calculate the complexity of the prompt based on the specified complexity type.
        """
        if self.score_name == "char_count_complexity":
            return len(text) / self.target_scale
        elif self.score_name == "word_count_complexity":
            return len(text.split()) / self.target_scale
        else:
            raise ValueError(f"Complexity type {self.complexity_type} not supported")
        