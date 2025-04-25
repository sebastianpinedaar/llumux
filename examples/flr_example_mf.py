import argparse

from flr.datasets.pairwise_dataset import PairwiseDataset
from flr.datasets.preprocessed_pairwise_dataset import PreprocessedPairwiseDataset
from flr.trainer import Trainer
from flr.trainer_args import TrainerArgs
from flr.scorers.pairwise.scorer import PairwiseScorer
from flr.scorers.matrix_factorization.scorer import MatrixFactorizationScorer
from flr.utils import LossTracker, CheckpointSaver


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
    parser.add_argument("--loss_fun", type=str, default="pairwise_logistic_loss", help="Loss function to use.")
    parser.add_argument("--max_length", type=int, default=512, help="Max length of the input sequence.")
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
    max_length = args.max_length

    # train_dataset = PreprocessedPairwiseDataset(dataset_name, split="train", test_size=test_size)
    # eval_dataset = PreprocessedPairwiseDataset(dataset_name, split="test", test_size=test_size)

    train_dataset = PairwiseDataset(dataset_name, split="train", test_size=test_size, random_sample=True)
    eval_dataset = PairwiseDataset(dataset_name, split="test", test_size=test_size, random_sample=True)

    model_list = train_dataset.collect_models()
    callbacks = [LossTracker(),
                 CheckpointSaver(workspace_path=workspace_path)]	
    
    scorer = MatrixFactorizationScorer(model_list, 
                                       prompt_embedder_name=prompt_embedder_name, 
                                       loss_fun=loss_fun,
                                       max_length=max_length)
    trainer_args = TrainerArgs(batch_size=batch_size, lr=lr, epochs=epochs)
    trainer = Trainer(scorer, trainer_args, train_dataset=train_dataset, 
                      eval_dataset=eval_dataset,
                     callbacks=callbacks)
    trainer.train()
    print("Done.")