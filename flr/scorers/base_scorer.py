from ..hub.model_hub import ModelHub

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


