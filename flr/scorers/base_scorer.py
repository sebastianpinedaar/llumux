import torch
import torch.nn as nn
from ..hub.model_hub import ModelHub

LAST_HIDDEN_DIM = {
    'bert-base-uncased': 768,
    'bert-base-cased': 768,
    'albert-base-v2': 768,	
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

    @classmethod
    def from_checkpoint(cls, path):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file {path} not found.")
        
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])  # instantiate with saved config
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
