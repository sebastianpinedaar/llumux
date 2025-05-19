from typing import List
import torch
import numpy as np
import yaml

from ..hub.model_hub import ModelHub
from ..scorers.base_scorer import BaseScorer
from ..routers.constant_router import ConstantRouter

class RandomRouter(ConstantRouter):
    """
    A router that uses the ratio of the scores from different scorers to make the final decision.
    The scorers might assess different aspects of the model such as
    efficiency, accuracy, fairness, etc.
    """
    def __init__(self, 
                 pick_largest: bool = True,
                 **kwargs):
        
        super().__init__(default_model_id=0, **kwargs)
    
    def route(self, prompt: str, model_list: List[str] = None):
        selecte_models = np.random.choice(len(self.model_info), size=len(prompt), replace=True)
        return selecte_models