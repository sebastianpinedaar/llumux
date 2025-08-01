import torch
from torch.utils.data import DataLoader

class RouterEvaluator:
    def __init__(self, router, evaluator_args):
        self.router = router
        self.evaluator_args = evaluator_args

    def evaluate(self, eval_dataset):
        eval_data_loader = DataLoader(eval_dataset, 
                                       batch_size=self.evaluator_args.batch_size, 
                                       shuffle=False)
        complexity = 0
        score = 0
        num_samples = 0
        for batch in eval_data_loader:
            temp_complexity, temp_score = self.score_batch(batch)
            num_samples += len(batch["prompts"])
            complexity += temp_complexity
            score += temp_score
        return complexity / num_samples, score / num_samples
    
    def score_batch(self, batch):
        selected_models = self.router.route(batch["prompts"])
        answers_complexity = []
        answers_score = []
        for i in range(len(batch["prompts"])):
            answer = batch["candidates"][selected_models[i]]["text"][i]
            score = batch["candidates"][selected_models[i]]["scores"][self.evaluator_args.score_name][i].item()
            answers_complexity.append(self.router.compute_complexity(answer))
            answers_score.append(score)

        return sum(answers_complexity), sum(answers_score) 