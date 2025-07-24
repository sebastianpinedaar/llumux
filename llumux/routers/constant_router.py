from typing import List
import torch
import numpy as np
import yaml

from .base_router  import BaseRouter
from ..hub.model_hub import ModelHub

class ConstantRouter(BaseRouter):
    """
    A router that uses the ratio of the scores from different scorers to make the final decision.
    The scorers might assess different aspects of the model such as
    efficiency, accuracy, fairness, etc.
    """
    def __init__(self, 
                 default_model_name: int = 0,
                 **kwargs):
        self.default_model_name = default_model_name
        self.kwargs = kwargs
        super().__init__(**kwargs)
        

    def route(self, prompts: List[str]) -> List[str]:
        selected_models = [self.default_model_name]*len(prompts)
        return selected_models
    
    def compute_complexity(self, answer):
        return len(answer)