from typing import List
import torch
import numpy as np
import yaml

from ..hub.model_hub import ModelHub
from ..scorers.base_scorer import BaseScorer

class RatioRouter:
    """
    A router that uses the ratio of the scores from different scorers to make the final decision.
    The scorers might assess different aspects of the model such as
    efficiency, accuracy, fairness, etc.
    """
    def __init__(self, 
                 scorer: BaseScorer,
                 profiler: BaseScorer,
                 threshold: float = 0.5,
                 strength: float = 1,
                 device: str = "cuda",
                 **kwargs):
        self.scorer = scorer.to(device)
        self.profiler = profiler.to(device)
        self.model_list = self.scorer.model_list
        self.threshold = threshold
        self.strength = strength
        self.kwargs = kwargs
        self.scorer.eval()
        self.profiler.eval()
        
        #read yaml
        with open("config/llm_instruct_models.yml", "r") as f:
            model_info = yaml.safe_load(f)
        self.model_size = np.array([int(model_info[model]) for model in self.model_list]).reshape(1, -1)

    def compute_assignment(self, profiler_out, scorer_out):
        
        prompt_complexity = profiler_out * self.model_size
        return scorer_out / (prompt_complexity**self.strength)
    
    def compute_complexity(self, answer):
        return len(answer)
    
    def route(self, prompt: str, model_list: List[str] = None) -> List[str]:

        num_samples = len(prompt)
        results_profiler = []
        results_scorer = []
        with torch.no_grad():
            for model_name in model_list:
                model_candidates= [model_name] * num_samples
                input = {
                    "prompt": prompt,
                    "model": model_candidates
                }
                profiler_out = self.profiler(**input)[0].cpu().numpy().reshape(-1, 1)
                scorer_out = self.scorer(**input).cpu().numpy().reshape(-1, 1)
                results_profiler.append(profiler_out)
                results_scorer.append(scorer_out)
        results_profiler = np.concatenate(results_profiler, axis=1)
        results_scorer = np.concatenate(results_scorer, axis=1)
        assignment = self.compute_assignment(results_scorer, results_profiler)
        selected_models = np.argmax(assignment, axis=1)
        #selected_models = np.array([self.model_list[i] for i in best_model])

        return selected_models