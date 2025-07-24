from .base_dataset import BaseDataset

class RouterDataset(BaseDataset):
    """Dataset class for routing tasks."""
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 seed: int = 42,
                 random_sample: bool = False,
                 fixed_len_train: int = 10000,
                 fixed_len_eval: int = 1000,
                 model_hub_name: str = None,
                 dataset_path: str = None,
                 **kwargs):
        super().__init__(dataset_name=dataset_name, 
                         split=split, 
                         test_size=test_size, 
                         seed=seed, 
                         random_sample=random_sample, 
                         fixed_len_train=fixed_len_train,
                         fixed_len_eval=fixed_len_eval,
                         model_hub_name=model_hub_name,
                         dataset_path=dataset_path)
    
    def process_candidates(self, candidates):
        new_candidates = {}
        for candidate in candidates:
            model = candidate["model"]
            new_candidates[model] = {
                "text" : candidate["text"],
                "scores" : candidate["scores"]
            }
        return new_candidates

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model: the model to compare
        - prompt: the prompt to compare the models
        - score: the score of the model
        """

        if self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            raise NotImplementedError("PointwiseDataset is not implemented for lmarena-ai/arena-human-preference-55k dataset")
        
        elif self.dataset_name == "llm-blender/mix-instruct":
            item = self.dataset[idx]
            item["candidates"] = self.process_candidates(item["candidates"])
            item.update({
                "prompts": item["instruction"] + ". "+ item["input"]
                }
            )
        elif self.dataset_name == "custom":
            item = self.dataset[idx]
            item["prompts"]= item.pop("prompt")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item