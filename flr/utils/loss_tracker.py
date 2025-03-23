import numpy as np
import logging

# use logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LossTracker:
    def __init__(self, freq=1, rolling_window=100):
        self.freq = freq
        self.rolling_window = rolling_window
        self.losses = []

    def on_batch_end(self, iteration_id, loss, **kwargs):
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
