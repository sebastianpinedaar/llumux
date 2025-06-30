from flr.datasets.pairwise_dataset import PairwiseDataset
from flr.trainer import Trainer
from flr.trainer_args import TrainerArgs
from flr.scorers.pairwise_scorer import PairwiseScorer
from flr.utils import LossTracker, CheckpointSaver
from flr.utils.parse_args import parse_args

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

    train_dataset = PairwiseDataset(dataset_name, split="train", test_size=test_size)
    eval_dataset = PairwiseDataset(dataset_name, split="test", test_size=test_size)

    model_list = train_dataset.collect_models()
    callbacks = [LossTracker(),
                 CheckpointSaver(workspace_path=workspace_path)]	
    
    scorer = PairwiseScorer(model_list, prompt_embedder_name=prompt_embedder_name, 
                            loss_fun_name=loss_fun_name,
                            hidden_size=hidden_size)
    trainer_args = TrainerArgs(batch_size=batch_size, lr=lr, epochs=epochs)
    trainer = Trainer(scorer, trainer_args, train_dataset=train_dataset, 
                    eval_dataset=eval_dataset,
                    callbacks=callbacks,
                     )
    trainer.train()
    print("Done.")