from llumux.trainer import Trainer
from llumux.trainer_args import TrainerArgs
from llumux.scorers.general_scorer import GeneralScorer
from llumux.datasets.listwise_dataset import ListwiseDataset
from llumux.callbacks import LossTracker, CheckpointSaver
from llumux.utils.parse_args import parse_args
from llumux.hub.model_hub import ModelHub

if __name__ == "__main__":

    args = parse_args()
    assert args.list_size > 0, "List size must be greater than 0."
    assert args.model_hub_name is not None, "Model hub name must be provided."

    train_dataset = ListwiseDataset(args.dataset_name, split="train", 
                                    test_size=args.test_size, 
                                    list_size=args.list_size)
    
    eval_dataset = ListwiseDataset(args.dataset_name, split="test", 
                                   test_size=args.test_size, 
                                   list_size=args.list_size)

    model_hub = ModelHub(args.model_hub_name)
    model_list = model_hub.get_models()
    
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