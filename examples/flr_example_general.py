import argparse

from flr.trainer import Trainer
from flr.trainer_args import TrainerArgs
from flr.scorers.general_scorer import GeneralScorer
from flr.datasets.listwise_dataset import ListwiseDataset
from flr.utils import LossTracker, CheckpointSaver
from flr.utils.parse_args import parse_args

if __name__ == "__main__":

    args = parse_args()
    train_dataset = ListwiseDataset(args.dataset_name, split="train", 
                                    test_size=args.test_size, 
                                    list_size=args.list_size)
    
    eval_dataset = ListwiseDataset(args.dataset_name, split="test", 
                                   test_size=args.test_size, 
                                   list_size=args.list_size)

    model_list = train_dataset.collect_models()
    callbacks = [LossTracker(),
                 CheckpointSaver(workspace_path=args.workspace_path)]	
    
    scorer = GeneralScorer(model_list, prompt_embedder_name=args.prompt_embedder_name, 
                            loss_fun_name=args.loss_fun_name,
                            hidden_size=args.hidden_size)
    
    trainer_args = TrainerArgs(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)
    trainer = Trainer(scorer, trainer_args, train_dataset=train_dataset, 
                    eval_dataset=eval_dataset,
                    callbacks=callbacks,
                    )
    trainer.train()
    print("Done.")