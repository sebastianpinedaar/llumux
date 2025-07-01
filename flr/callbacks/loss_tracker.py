import numpy as np
import logging

from .base_callback import BaseCallback

# use logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LossTracker(BaseCallback):
    def __init__(self, freq: int =1, 
                 rolling_window: int =100):
        self.freq = freq
        self.rolling_window = rolling_window
        self.losses = []
    
    def start(self, scorer_name):
        self.scorer_name = scorer_name

    def on_batch_end(self, iteration_id, loss, **kwargs):
        assert self.scorer_name is not None, "Need to start callback with the model name."
        self.losses.append(loss.item())
        if len(self.losses) % self.freq == 0:
            #rollwing window average
            rolling_window = self.rolling_window
            if len(self.losses) > rolling_window:
                rolling_window_loss = np.mean(self.losses[-rolling_window:])
            else:
                rolling_window_loss = np.mean(self.losses)
            logger.info(f"Iteration {iteration_id}, batch loss: {loss.item()}")
            logger.info(f"Iteration {iteration_id}, rolling window loss: {rolling_window_loss}")
