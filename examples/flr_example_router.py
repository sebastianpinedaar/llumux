from pathlib import Path
from llumux.scorers.pointwise_scorer import PointwiseScorer
from llumux.datasets.router_dataset import RouterDataset
from llumux.routers.ratio_router import RatioRouter
from llumux.routers.greedy_router import GreedyRouter
from llumux.routers.random_router import RandomRouter
from llumux.router_evaluator import RouterEvaluator
from llumux.router_evaluator_args import RouterEvaluatorArgs
from llumux.hub.model_hub import ModelHub
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
    model_hub_name = args.model_hub_name

    eval_dataset = RouterDataset(dataset_name, split="test", 
                                                model_hub_name=model_hub_name,
                                                test_size=test_size)

    model_hub = ModelHub(model_hub_name)
    perf_scorer = PointwiseScorer.from_checkpoint(Path("workspace/scorers/perf_scorer/best_checkpoint.pt"))
    cost_scorer = PointwiseScorer.from_checkpoint(Path("workspace/scorers/cost_scorer/best_checkpoint.pt"))

    evaluator_args = RouterEvaluatorArgs(batch_size=batch_size)
    for threshold in [0, 0.25, 0.5]:
        print(f"Evaluating with threshold {threshold}")
        
        scorers = {
            "perf_scorer": perf_scorer,
            "cost_scorer": cost_scorer
        }
        router = RatioRouter(scorers=scorers, model_hub=model_hub, threshold=threshold)
        evaluator = RouterEvaluator(router=router, evaluator_args=evaluator_args)
        eval_score = evaluator.evaluate(eval_dataset=eval_dataset)
        print(f"Eval score: {eval_score}")

    router = GreedyRouter(pick_largest=True)
    evaluator = RouterEvaluator(router=router, evaluator_args=evaluator_args)
    eval_score = evaluator.evaluate(eval_dataset=eval_dataset)
    print(f"Eval score: {eval_score}")

    router = GreedyRouter(pick_largest=False)
    evaluator = RouterEvaluator(router=router, evaluator_args=evaluator_args)
    eval_score = evaluator.evaluate(eval_dataset=eval_dataset)
    print(f"Eval score: {eval_score}")

    router = RandomRouter()
    evaluator = RouterEvaluator(router=router, 
                                 evaluator_args=evaluator_args)
    eval_score = evaluator.evaluate(eval_dataset=eval_dataset)
    print(f"Eval score: {eval_score}")
    print("Done")