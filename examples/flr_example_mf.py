from flr.datasets.pairwise_dataset import PairwiseDataset
from flr.datasets.preprocessed_pairwise_dataset import PreprocessedPairwiseDataset
from flr.trainer import Trainer
from flr.trainer_args import TrainerArgs
from flr.scorers.pairwise.scorer import PairwiseScorer
from flr.scorers.matrix_factorization.scorer import MatrixFactorizationScorer
from flr.utils import LossTracker, CheckpointSaver

if __name__ == "__main__":

    workspace_path = "workspaces/test"
    dataset_name = "lmarena-ai/arena-human-preference-55k-preprocessed"
    test_size = 0.001
    lr = 0.001
    batch_size = 128
    epochs = 20
    prompt_embedder_name = "identity"

    train_dataset = PreprocessedPairwiseDataset(dataset_name, split="train", test_size=test_size)
    eval_dataset = PreprocessedPairwiseDataset(dataset_name, split="test", test_size=test_size)


    model_list = train_dataset.collect_models()
    callbacks = [LossTracker(),
                 CheckpointSaver(workspace_path=workspace_path)]	
    
    scorer = MatrixFactorizationScorer(model_list, prompt_embedder_name=prompt_embedder_name)
    trainer_args = TrainerArgs(batch_size=batch_size, lr=lr, epochs=epochs)
    trainer = Trainer(scorer, trainer_args, train_dataset=train_dataset, 
                      eval_dataset=eval_dataset,
                     callbacks=callbacks)
    trainer.train()
    print("Done.")