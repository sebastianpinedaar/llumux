from typing import Dict, List
import numpy as np

from ..scorers.base_scorer import BaseScorer
from ..hub.model_hub import ModelHub

class BaseRouter:   
    """
    A router combines the scorers to make the final decision.
    The scorers might assess different aspects of the model such as
    efficiency, accuracy, fairness, etc.
    """
    def __init__(self, 
                    scorers: Dict[str, BaseScorer],	
                    model_hub: ModelHub,
                    threshold: float = 0.5,
                    top_k: int = 1,
                   **kwargs):
        self.scorers = scorers
        self.top_k = top_k
        self.threshold = threshold
        self.model_hub = model_hub
        self.model_size = np.array(
                        self.model_hub.get_attributes_from_model_card("model_size")
                        ).reshape(1, -1)
        self.models = self.model_hub.get_models()

    def route(self, prompts: List[str]) -> List[str]:
        """
        Route the prompt to the top k models based on the scorers.
        Args:
            prompt: A prompt string
        Returns:
            A list of top k models
            """
        raise NotImplementedError
    
    

