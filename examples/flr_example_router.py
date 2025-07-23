import argparse
from pathlib import Path
from llumux.scorers.pointwise_scorer import PointwiseScorer
from llumux.datasets.pointwise_dataset import PointwiseDataset
from llumux.scorers.matrix_factorization_scorer import MatrixFactorizationScorer
from llumux.datasets.text_complexity_dataset import PromptComplexityDataset
from llumux.datasets.router_dataset import RouterDataset
from llumux.routers.ratio_router import RatioRouter
from llumux.routers.greedy_router import GreedyRouter
from llumux.routers.random_router import RandomRouter
from llumux.router_evaluator import RouterEvaluator
from llumux.router_evaluator_args import RouterEvaluatorArgs
from llumux.utils.parse_args import parse_args

if __name__ == "__main__":

    args = parse_args()
    workspace_path = args.workspace_path
    dataset_name = args.dataset_name
    test_size = args.test_size
    prompt_embedder_name = args.prompt_embedder_name
    lr = args.lr
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    epochs = args.epochs
    loss_fun_name = args.loss_fun_name
    train_dataset = PointwiseDataset(dataset_name, split="train", test_size=test_size, random_sample=True)
    model_list = train_dataset.collect_models()
    eval_dataset = RouterDataset(dataset_name, split="test", 
                                                model_list=model_list,
                                                test_size=test_size)


    #scorer = PointwiseScorer.from_checkpoint(Path("workspace/complexity_pred/checkpoints/best_checkpoint.pt"))
    profiler = PointwiseScorer.from_checkpoint(Path("workspace/pref_pred/checkpoints/best_checkpoint.pt"))
    scorer = MatrixFactorizationScorer.from_checkpoint(Path("workspace/test/checkpoints/best_checkpoint.pt"))

    evaluator_args = RouterEvaluatorArgs(batch_size=batch_size)

    for strength in [0, 0.25, 0.5]:
        print(f"Evaluating with strength {strength}")
        
        scorers = {
            "perf_scorer": scorer,
            "cost_scorer": profiler
        }
        router = RatioRouter(scorer=scorer, profiler=profiler, strength=strength)
        evaluator = RouterEvaluator(router=router, 
                                     evaluator_args=evaluator_args, 
                                     eval_dataset=eval_dataset)
        
        eval_score = evaluator.evaluate()
        print(f"Eval score: {eval_score}")

    router = GreedyRouter(pick_largest=True)
    evaluator = RouterEvaluator(router=router, 
                                 evaluator_args=evaluator_args, 
                                 eval_dataset=eval_dataset)
    eval_score = evaluator.evaluate()
    print(f"Eval score: {eval_score}")

    router = GreedyRouter(pick_largest=False)
    evaluator = RouterEvaluator(router=router, 
                                 evaluator_args=evaluator_args, 
                                 eval_dataset=eval_dataset)
    eval_score = evaluator.evaluate()
    print(f"Eval score: {eval_score}")


    router = RandomRouter()
    evaluator = RouterEvaluator(router=router, 
                                 evaluator_args=evaluator_args, 
                                 eval_dataset=eval_dataset)
    eval_score = evaluator.evaluate()
    print(f"Eval score: {eval_score}")
    print("Done")