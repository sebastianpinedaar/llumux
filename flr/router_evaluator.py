import torch
from torch.utils.data import DataLoader

class RouterEvaluator:
    def __init__(self, router, evaluator_args, eval_dataset):
        self.router = router
        self.evaluator_args = evaluator_args
        self.eval_dataset = eval_dataset
        self.eval_data_loader = DataLoader(self.eval_dataset, 
                                       batch_size=self.evaluator_args.batch_size, 
                                       shuffle=False)

    def evaluate(self):
        complexity = 0
        score = 0
        num_samples = 0
        for batch in self.eval_data_loader:
            temp_complexity, temp_score = self.score_batch(batch)
            num_samples += len(batch["prompt"])
            complexity += temp_complexity
            score += temp_score
        return complexity / num_samples, score / num_samples
    
    def score_batch(self, batch):
        models = [x["model"][0] for x in batch["candidates"][0]]
        selected_models = self.router.route(batch["prompt"], models)
        answers_complexity = []
        answers_score = []
        for i in range(len(batch["prompt"])):
            answer = batch["candidates"][0][selected_models[i]]["text"][i]
            score = batch["candidates"][0][selected_models[i]]["scores"]["bertscore"][i].item()
            answers_complexity.append(self.router.compute_complexity(answer))
            answers_score.append(score)

        return sum(answers_complexity), sum(answers_score) 