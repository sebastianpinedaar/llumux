from ..hub.model_hub import ModelHub

LAST_HIDDEN_DIM = {
    'bert-base-uncased': 768,
    'bert-base-cased': 768,
    'albert-base-v2': 768,	
    'identity': 768
}

class BaseScorer:   
    def __init__(self, 
                    model_hub: ModelHub,
                   **kwargs):
        self.model_hub = model_hub

    def score(self, prompt: str):
        """
        Args:
            prompt: A prompt string
        Returns:
            A list of scores
            """
        raise NotImplementedError


