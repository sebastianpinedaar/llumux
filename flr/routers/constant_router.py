from typing import List
import torch
import numpy as np
import yaml

from ..hub.model_hub import ModelHub
from ..scorers.base_scorer import BaseScorer

class ConstantRouter:
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
        
        with open("config/llm_instruct_models.yml", "r") as f:
            self.model_info = yaml.safe_load(f)
        self.model_size = np.array(list(self.model_info.values())).reshape(1, -1)


    def route(self, prompt: str, model_list: List[str] = None):
        ix =np.where(np.array(model_list) == self.default_model_name)[0]
        selected_models = np.array([ix]*len(prompt) ).reshape(-1)
        return selected_models
    
    def compute_complexity(self, answer):
        return len(answer)