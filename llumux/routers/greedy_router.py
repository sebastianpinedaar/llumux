from typing import List
import torch
import numpy as np
import yaml

from ..hub.model_hub import ModelHub
from ..scorers.base_scorer import BaseScorer
from ..routers.constant_router import ConstantRouter

class GreedyRouter(ConstantRouter):
    """
    A router that uses the ratio of the scores from different scorers to make the final decision.
    The scorers might assess different aspects of the model such as
    efficiency, accuracy, fairness, etc.
    """
    def __init__(self, 
                 pick_largest: bool = True,
                 **kwargs):
        
        super().__init__(default_model_id=0, **kwargs)
        
        
        if pick_largest:
            picking_size = max(self.model_size[0]).item()
        else:
            picking_size = min(self.model_size[0]).item()
        self.default_model_id = np.random.choice(np.where(self.model_size.flatten() == picking_size)[0]).item()
        self.default_model_name = self.models[self.default_model_id]