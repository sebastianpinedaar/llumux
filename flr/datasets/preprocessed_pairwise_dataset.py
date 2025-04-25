
from datasets import load_dataset,  load_from_disk
import torch
import numpy as np
from transformers import BertModel, BertTokenizer

from pathlib import Path
from .base_dataset import BaseDataset

NUMBER_OF_MODELS = {
    "lmarena-ai/arena-human-preference-55k": 64
}


def preprocess(item, dataset):
    new_item = {}
    new_item["prompt"] = dataset.get_embeddings(item["prompt"])  
    new_item["model_a"] = item["model_a"]
    new_item["model_b"] = item["model_b"]
    new_item["target"] = [x_a-1*x_b for x_a, x_b in zip(item["winner_model_a"], item["winner_model_b"])]
    return new_item

class PreprocessedPairwiseDataset(BaseDataset):
    def __init__(self, dataset_name: str, 
                 split: str = "train", 
                 test_size: float = 0.1,
                 prompt_embedder_name="bert-base-uncased",
                 seed: int = 42,
                 device = "cuda"):
        self.dataset_name = dataset_name
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.model_count = None
        self.device = device
        self.prompt_embedder = BertModel.from_pretrained(prompt_embedder_name, 
                                                         torch_dtype=torch.float32, 
                                                         attn_implementation="sdpa").to(device)
        self.prompt_tokenizer = BertTokenizer.from_pretrained(prompt_embedder_name)
 
        if dataset_name == "lmarena-ai/arena-human-preference-55k":
            dataset_before_split = load_dataset(dataset_name)["train"]
            dataset = dataset_before_split.train_test_split(test_size=test_size, seed=seed)[split]
            self.dataset = dataset.map(lambda x: preprocess(x, self), batched=True, batch_size=32)
            self.dataset.save_to_disk("workspace/"+ dataset_name + f"-preprocessed-{split}")
        elif dataset_name == "lmarena-ai/arena-human-preference-55k-preprocessed":
            self.dataset = load_from_disk("workspace/"+dataset_name+"-"+split)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")



    def get_embeddings(self, prompt):
        with torch.no_grad():
            tokens = self.prompt_tokenizer(prompt, return_tensors='pt', 
                                        padding="max_length", 
                                        max_length=512, 
                                        truncation=True).to(self.device)
            input_ids, token_type_ids, attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
            prompt_embedding = self.prompt_embedder(input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)
            prompt_embedding = prompt_embedding.last_hidden_state[:, 0, :]
        return prompt_embedding.detach().cpu().numpy().tolist()

    def get_number_of_models(self):
        return NUMBER_OF_MODELS[self.dataset_name]
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing the following keys:
        - model_a: the first model to compare
        - model_b: the second model to compare
        - prompt: the prompt to compare the models
        - winner_model_a: 1 if model_a is better, -1 if model_b is better
        """

        if self.dataset_name == "lmarena-ai/arena-human-preference-55k":
            item = self.dataset[idx]
            item = { 
                "prompt": item["prompt"], \
                "target": item["winner_model_a"]-1*item["winner_model_b"], \
                "model_a": item["model_a"],
                "model_b": item["model_b"]
            }
        elif self.dataset_name == "lmarena-ai/arena-human-preference-55k-preprocessed":
            item = self.dataset[idx]
            item = { 
                "prompt": item["prompt"], \
                "target": item["target"], \
                "model_a": item["model_a"],
                "model_b": item["model_b"]
            }         
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return item

    def collect_models(self):
        if self.model_list is None:
            models = []
            for item in iter(self):
                models.append(item["model_a"])
                models.append(item["model_b"])
            
            self.model_list = np.unique(models).tolist()

        return self.model_list


