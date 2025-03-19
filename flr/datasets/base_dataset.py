class BaseDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        # Implement this method to return a dictionary containing the following
        raise NotImplementedError