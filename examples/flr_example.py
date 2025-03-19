from flr.datasets.pairwise_dataset import PairwiseDataset
from flr.routers.bert_pairwise.model import BertPairwiseScorer
from flr.trainer import Trainer

if __name__ == "__main__":
    dataset = PairwiseDataset("lmarena-ai/arena-human-preference-55k", split="train", test_size=0.1, seed=1)
    trainer = Trainer(model, dataset)