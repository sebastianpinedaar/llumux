from flr.datasets.preprocessed_pairwise_dataset import PreprocessedPairwiseDataset

dataset = PreprocessedPairwiseDataset("lmarena-ai/arena-human-preference-55k", split="train", test_size=0.98, seed=1)
print(dataset.get_number_of_models())