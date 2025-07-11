from typing import List

from .general_scorer import GeneralScorer

class PairwiseScorer(GeneralScorer):
    def __init__(self, *args, **kwargs):
        super(PairwiseScorer, self).__init__(*args, **kwargs)

    def forward(self, prompts: List[str], 
                    models_a: List[str] = None, 
                    models_b: List[str] = None,
                    models: List[str] = None,
                    targets: List[str] = None, 
                **kwargs):

        if targets is not None:
            prompt_embeddings = self.get_prompt_embedding(prompts)
            prompt_embeddings = self.ln1(prompt_embeddings)
            prompt_embeddings = self.fc1_prompt(prompt_embeddings)

            score_a = self.score(prompt_embeddings, models_a)
            score_b = self.score(prompt_embeddings, models_b)
            loss = self.loss_fn((score_a, score_b), targets.to(self.device))

            return [score_a, score_b], loss

        else:
            assert models is not None, "Models cannnot be not for individual evaluation (when targets is none)"
            return super(PairwiseScorer, self).forward(prompts, models)



