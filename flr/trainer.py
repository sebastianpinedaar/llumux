import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, trainer_args, train_dataset, callbacks):
        self.model = model
        self.trainer_args = trainer_args
        self.train_dataset = train_dataset
        self.callbacks = callbacks
        self.data_loader = DataLoader(self.train_dataset, 
                                      batch_size=self.trainer_args.batch_size, 
                                      shuffle=True)
    
    def train(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.trainer_args.lr)

        for epoch in range(self.trainer_args.epochs):
            for batch in self.data_loader:
                self.model.train()
                self.model.zero_grad()
                loss = self.model(**batch)
                loss.backward()
                self.optimizer.step()
                for callback in self.callbacks:
                    callback.on_batch_end()
            for callback in self.callbacks:
                callback.on_epoch_end()