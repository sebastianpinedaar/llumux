from typing import List

from ..hub.model_hub import ModelHub
from ..scorers.base_scorer import BaseScorer

class BaseRouter:   
    """
    A router combines the scorers to make the final decision.
    The scorers might assess different aspects of the model such as
    efficiency, accuracy, fairness, etc.
    """
    def __init__(self, 
                    scorers: List[BaseScorer],	
                    top_k: int = 1,
                    threshold: float = 0.5,
                   **kwargs):
        self.scorers = scorers
        self.top_k = top_k
        self.threshold = threshold

    def route(self, prompt: str):
        """
        Args:
            prompt: A prompt string
        Returns:
            A list of top k models
            """
        raise NotImplementedError
    
    

