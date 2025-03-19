from ..hub.model_hub import ModelHub

class BaseRouter:   
    def __init__(self, 
                    model_hub: ModelHub,
                    top_k: int = 1,
                   **kwargs):
        self.model_hub = model_hub
        self.top_k = top_k

    def route(self, prompt: str):
        """
        Args:
            prompt: A prompt string
        Returns:
            A list of top-k model names
            """
        raise NotImplementedError
    
    def train(self, dataset: Dataset):
        """
        Args:
            dataset: A dataset object
        Returns:
            None
        """
        raise NotImplementedError

