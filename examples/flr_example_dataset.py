from flr.datasets.preprocessed_pairwise_dataset import PreprocessedPairwiseDataset
from flr.datasets.pairwise_dataset import PairwiseDataset

dataset = PairwiseDataset("llm-blender/mix-instruct", split="train", test_size=0.02, seed=1, random_sample=True)

[print(dataset[i]) for i in range(len(dataset))]
dataset = PreprocessedPairwiseDataset("lmarena-ai/arena-human-preference-55k", split="test", test_size=0.02, seed=1)
print(dataset.get_number_of_models())


