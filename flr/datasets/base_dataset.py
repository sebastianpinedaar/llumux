import numpy as np
from datasets import load_dataset

class BaseDataset:
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 fixed_len_train: int = 10000,
                 fixed_len_eval: int = 1000):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.random_sample = random_sample
        self.fixed_len_train = fixed_len_train
        self.fixed_len_eval = fixed_len_eval
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
            dataset = load_dataset(dataset_name)[split]
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        return dataset
    
    def __len__(self):
        if self.random_sample:
            return self.fixed_len_train if self.split == "train" else self.fixed_len_eval
        elif self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            return len(self.dataset)
        elif self.dataset_name == "llm-blender/mix-instruct":
            return len(self.dataset)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in the subclass")