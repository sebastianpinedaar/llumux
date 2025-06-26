import argparse
from pathlib import Path
from flr.scorers.pointwise_scorer import PointwiseScorer
from flr.scorers.complexity_scorer import ComplexityPredictor
from flr.datasets.pointwise_dataset import PointwiseDataset
from flr.scorers.matrix_factorization_scorer import MatrixFactorizationScorer
from flr.datasets.prompt_complexity_dataset import PromptComplexityDataset
from flr.datasets.route_dataset import RouteDataset
from flr.routers.ratio_router import RatioRouter
from flr.routers.greedy_router import GreedyRouter
from flr.routers.random_router import RandomRouter
from flr.router_evaluator import RouterEvaluator
from flr.router_evaluator_args import RouterEvaluatorArgs

def parse_args():
    parser = argparse.ArgumentParser(description="Train a pairwise ranking model.")
    parser.add_argument("--workspace_path", type=str, default= "workspaces/test", help="Path to save the workspace.")
    parser.add_argument("--dataset_name", type=str,default= "lmarena-ai/arena-human-preference-55k", help="Name of the dataset.")
    parser.add_argument("--test_size", type=float, default=0.001, help="Size of the test set.")
    parser.add_argument("--prompt_embedder_name", type=str, default="identity", help="Name of the prompt embedder.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of the model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train.")
    parser.add_argument("--loss_fun", type=str, default="mse", help="Loss function to use.")

    return parser.parse_args()

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
    loss_fun = args.loss_fun
    train_dataset = PointwiseDataset(dataset_name, split="train", test_size=test_size, random_sample=True)
    model_list = train_dataset.collect_models()
    eval_dataset = RouteDataset(dataset_name, split="test", 
                                                model_list=model_list,
                                                test_size=test_size)


    #scorer = PointwiseScorer.from_checkpoint(Path("workspace/complexity_pred/checkpoints/best_checkpoint.pt"))
    profiler = PointwiseScorer.from_checkpoint(Path("workspace/pref_pred/checkpoints/best_checkpoint.pt"))
    scorer = MatrixFactorizationScorer.from_checkpoint(Path("workspace/test/checkpoints/best_checkpoint.pt"))

    evaluator_args = RouterEvaluatorArgs(batch_size=batch_size)

    for strength in [0, 0.25, 0.5]:
        print(f"Evaluating with strength {strength}")
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