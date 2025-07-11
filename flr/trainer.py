import torch
import logging
from torch.utils.data import DataLoader

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, trainer_args, train_dataset,
                  eval_dataset=None,
                  callbacks: list = None):
        self.model = model
        self.trainer_args = trainer_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks or []	
        self.train_data_loader = DataLoader(self.train_dataset, 
                                      batch_size=self.trainer_args.batch_size, 
                                      shuffle=True)
        
        if self.eval_dataset is not None:
            self.eval_data_loader = DataLoader(self.eval_dataset, 
                                      batch_size=self.trainer_args.batch_size, 
                                      shuffle=False)
        
    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.trainer_args.lr)

        self.model.train()
        #self.model.to(self.trainer_args.device)
        eval_loss = np.inf
        iteration_id = 0
        for epoch in range(self.trainer_args.epochs):
            for batch in self.train_data_loader:
                self.model.zero_grad()
                _, loss = self.model(**batch)
                loss.backward()
                optimizer.step()

                iteration_id += 1

                if iteration_id % self.trainer_args.eval_freq == 0:
                    eval_loss = self.evaluate()
                    self.model.train()

                for callback in self.callbacks:
                    callback.on_batch_end(iteration_id=iteration_id, 
                                          loss=loss, 
                                          eval_loss=eval_loss,
                                          model=self.model,
                                          batch=batch)


    def evaluate(self):
        
        self.model.eval()
        losses = []	
        with torch.no_grad():
            for batch_id, batch in enumerate(self.eval_data_loader):
                _, loss = self.model(**batch)
                losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        logger.info(f"Avg. eval loss: {avg_loss}")
        
        return avg_loss

