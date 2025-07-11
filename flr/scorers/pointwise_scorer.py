from typing import List

from .general_scorer import GeneralScorer

class PointwiseScorer(GeneralScorer):
    def __init__(self, *args, **kwargs):
        super(PointwiseScorer, self).__init__(*args, **kwargs)

    def forward(self, prompts: List[str], 
                models: List[str], 
                targets: List[float] = None, **kwargs):
        
        if targets is not None:
            prompt_embeddings = self.get_prompt_embedding(prompts)
            prompt_embeddings = self.ln1(prompt_embeddings)
            prompt_embeddings = self.fc1_prompt(prompt_embeddings)
            score = self.score(prompt_embeddings, models)
            loss = self.loss_fn(score.reshape(-1), targets.to(self.device).float())

            return score, loss
        
        else:
        
            return super(PointwiseScorer, self).forward(prompts, models)