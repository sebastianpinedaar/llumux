import torch
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointSaver:
    def __init__(self, 
                  workspace_path=None,
                  freq=1,
                  checkpoint_name="checkpoint.pt",
                  checkpoint_dir="checkpoints"):
        
        assert workspace_path is not None, "Workspace path is required."
        self.workspace_path = Path(workspace_path)
        self.freq = freq
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = checkpoint_dir
        self.full_checkpoint_dir = self.workspace_path / checkpoint_dir
        self.full_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = np.inf

    def on_batch_end(self, iteration_id, model, **kwargs):

        if iteration_id % self.freq == 0:
            torch.save({
                'config': model.get_config(),
                'model_state_dict': model.state_dict(),
            }, self.full_checkpoint_dir / self.checkpoint_name)

        if kwargs["eval_loss"] < self.best_loss:
            logger.info(f"Saving checkpoint for epoch {iteration_id}.")

            self.best_loss = kwargs["eval_loss"]
            torch.save({
                'config': model.get_config(),
                'model_state_dict': model.state_dict(),
            }, self.full_checkpoint_dir / ("best_" + self.checkpoint_name ))
     