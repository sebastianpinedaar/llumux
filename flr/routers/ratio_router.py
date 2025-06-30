from typing import List
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
                 scorers: List[BaseScorer],
                 model_hub_name: str = "llm_instruct_models",
                 strength: float = 1,
                 device: str = "cuda",
                 **kwargs):
        
        assert len(scorers) == 2, "RatioRouter requires exactly two scorers: performance and complexity cost."
        self.perf_scorer = scorers[0].to(device)
        self.cost_scorer = scorers[1].to(device)
        self.model_hub_name = model_hub_name
        self.strength = strength
        self.kwargs = kwargs
        self.perf_scorer.eval()
        self.cost_scorer.eval()
        self.model_hub = ModelHub(model_hub_name=model_hub_name)
        self.model_size = np.array(
                        self.model_hub.get_attributes_from_model_card("model_size")
                        ).reshape(1, -1)

    def compute_assignment(self, profiler_out, scorer_out):
        prompt_complexity = profiler_out * self.model_size
        return scorer_out / (prompt_complexity**self.strength)
    
    def compute_complexity(self, answer):
        return len(answer)
    
    def route(self, prompt: str) -> List[str]:
        num_samples = len(prompt)
        results_profiler = []
        results_scorer = []
        with torch.no_grad():
            for model_name in self.model_hub.get_models():
                model_candidates= [model_name] * num_samples
                input = {
                    "prompt": prompt,
                    "model": model_candidates
                }
                profiler_out = self.cost_scorer(**input)[0].cpu().numpy().reshape(-1, 1)
                scorer_out = self.perf_scorer(**input)[0].cpu().numpy().reshape(-1, 1)
                results_profiler.append(profiler_out)
                results_scorer.append(scorer_out)
        results_profiler = np.concatenate(results_profiler, axis=1)
        results_scorer = np.concatenate(results_scorer, axis=1)
        assignment = self.compute_assignment(results_scorer, results_profiler)
        selected_models = np.argmax(assignment, axis=1)

        return selected_models