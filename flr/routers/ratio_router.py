from typing import Dict, List
import torch
import numpy as np

from ..scorers.base_scorer import BaseScorer
from ..hub.model_hub import ModelHub

class RatioRouter:
    """
    A router that uses the ratio of the scores from different scorers to make the final decision.
    The scorers might assess different aspects of the model such as
    efficiency, accuracy, fairness, etc.
    """
    def __init__(self, 
                 scorers: Dict[str, BaseScorer],
                 model_hub: ModelHub,
                 strength: float = 1,
                 device: str = "cuda",
                 **kwargs):
        
        assert len(scorers) == 2, "RatioRouter requires exactly two scorers: performance and complexity cost."

        self.perf_scorer = scorers["perf_scorer"].to(device)
        self.cost_scorer = scorers["cost_scorer"].to(device)
        self.strength = strength
        self.kwargs = kwargs
        self.perf_scorer.eval()
        self.cost_scorer.eval()
        self.model_hub = model_hub
        self.model_size = np.array(
                        self.model_hub.get_attributes_from_model_card("model_size")
                        ).reshape(1, -1)
        self.models = self.model_hub.get_models()

    def compute_assignment(self, cost_scorer_out, perf_scorer_out):
        prompt_complexity = cost_scorer_out * self.model_size
        return perf_scorer_out / (prompt_complexity**self.strength)
    
    def compute_complexity(self, answer):
        return len(answer)
    
    def route(self, prompt: str) -> List[str]:
        num_samples = len(prompt)

        with torch.no_grad():
            model_candidates= [[model_name] * num_samples \
                            for model_name in self.models]
            input = {
                "prompt": prompt,
                "models": model_candidates
            }
            cost_scorer_out = self.cost_scorer(**input)[0].cpu().numpy()
            perf_scorer_out = self.perf_scorer(**input)[0].cpu().numpy()

        assignment = self.compute_assignment(cost_scorer_out, perf_scorer_out)
        selected_models = [self.models[i] for i in np.argmax(assignment, axis=1)]

        return selected_models