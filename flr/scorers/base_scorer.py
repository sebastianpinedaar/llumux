import torch
import torch.nn as nn

from typing import List
from transformers import BertModel, BertTokenizer
from transformers import AlbertTokenizer, AlbertModel
from transformers import DistilBertTokenizer, DistilBertModel

class BaseScorer(nn.Module):   
    def __init__(self, use_frozen_embedder: bool = False,
                   **kwargs):
        super(BaseScorer, self).__init__()
        self.use_frozen_embedder = use_frozen_embedder

    def score(self, prompt: str):
        """
        Args:
            prompt: A prompt string
        Returns:
            A list of scores
            """
        raise NotImplementedError

    def freeze_embedder(self):
        for param in self.prompt_embedder.parameters():
            param.requires_grad = False

    def get_prompt_embedding(self, prompts: List[str]):
        if self.prompt_embedder_name in ["bert-base-uncased", "albert-base-v2"]:
            tokens = self.prompt_tokenizer(prompts, return_tensors='pt', 
                                           padding="max_length", 
                                           max_length=self.max_length, 
                                           truncation=True).to(self.device)
            input_ids, token_type_ids, attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
            prompt_embedding = self.prompt_embedder(input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask)
            prompt_embedding = prompt_embedding.last_hidden_state[:, 0, :]
        elif self.prompt_embedder_name == "identity":
            prompt_embedding = torch.vstack(prompts).T.to(self.device).float()
        return prompt_embedding


    def initialize_prompt_embedder(self):
        if self.prompt_embedder_name == "bert-base-uncased":
            self.prompt_embedder = BertModel.from_pretrained(self.prompt_embedder_name, 
                                                            torch_dtype=torch.float32, 
                                                            attn_implementation="sdpa").to(self.device)
            self.prompt_tokenizer = BertTokenizer.from_pretrained(self.prompt_embedder_name)
        elif self.prompt_embedder_name == "albert-base-v2":
            self.prompt_embedder = AlbertModel.from_pretrained(self.prompt_embedder_name, 
                                                              torch_dtype=torch.float32).to(self.device)
            self.prompt_tokenizer = AlbertTokenizer.from_pretrained(self.prompt_embedder_name)
        
        elif self.prompt_embedder_name == "distilbert-base-cased":
            self.prompt_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.prompt_embedder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        else:
            raise ValueError(f"Prompt embedder {self.prompt_embedder_name} not supported")

        if self.use_frozen_embedder:
            self.freeze_embedder()

    @classmethod
    def from_checkpoint(cls, path):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file {path} not found.")
        
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])  # instantiate with saved config
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
