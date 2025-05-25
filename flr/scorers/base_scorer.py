import torch
import torch.nn as nn
from ..hub.model_hub import ModelHub

from transformers import BertModel, BertTokenizer
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import AlbertModel
from transformers import DistilBertTokenizer, DistilBertModel

LAST_HIDDEN_DIM = {
    'bert-base-uncased': 768,
    'bert-base-cased': 768,
    'albert-base-v2': 768,
    'distilbert-base-cased': 768,
    'prajjwal1/bert-tiny': 128,	
    'prajjwal1/bert-small': 256,
    'identity': 768
}

class BaseScorer(nn.Module):   
    def __init__(self, 
                   **kwargs):
        super(BaseScorer, self).__init__()

    def score(self, prompt: str):
        """
        Args:
            prompt: A prompt string
        Returns:
            A list of scores
            """
        raise NotImplementedError

    def get_prompt_embedding(self, prompt):
        if self.prompt_embedder_name in ["bert-base-uncased", "prajjwal1/bert-tiny", "albert-base-v2", "distilbert-base-cased"]:
            tokens = self.prompt_tokenizer(prompt, return_tensors='pt', padding="max_length", max_length=self.max_length, truncation=True).to(self.device)
            prompt_embedding = self.prompt_embedder(**tokens)
            prompt_embedding = prompt_embedding.last_hidden_state[:, 0, :]    
        elif self.prompt_embedder_name == "identity":
            prompt_embedding = torch.vstack(prompt).T.to(self.device).float()
        else: 
            raise ValueError(f"Prompt embedder {self.prompt_embedder_name} not supported")
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

    @classmethod
    def from_checkpoint(cls, path):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file {path} not found.")
        
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])  # instantiate with saved config
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
