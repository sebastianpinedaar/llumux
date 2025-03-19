from ..base_router import BaseRouter
from ..dataset import Dataset

class BertPairwiseRouter(BaseRouter):
    def __init__(self, model_hub, top_k: int = 1, **kwargs):
        super().__init__(model_hub, top_k, **kwargs)

    def route(self, prompt: str):
        #TODO: Implement routing logic
        return ["dummy_model"]
    
    def train(self, dataset: Dataset, trainer: Trainer):
        raise NotImplementedError
    
    def evaluate(self, dataset: Dataset, evaluator: Evaluator):
        raise NotImplementedError

    def __str__(self):
        return "BertPairwiseRouter"
