import os

import torch
import logging
import numpy as np

from pathlib import Path

from .base_callback import BaseCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointSaver(BaseCallback):
    def __init__(self, 
                  workspace_path: str = None,
                  freq: int = 1,
                  checkpoint_dir:str ="scorers"):
        
        if workspace_path is None:
            workspace_path = Path(os.environ.get("FLR_HOME", "")) / "workspace"
        self.workspace_path = Path(workspace_path) 
        self.freq = freq
        self.checkpoint_dir = checkpoint_dir

    def start(self, scorer_name):
        self.scorer_name = scorer_name
        self.full_checkpoint_dir = self.workspace_path / self.checkpoint_dir / self.scorer_name
        self.full_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = np.inf

    def on_batch_end(self, iteration_id, model, **kwargs):
        assert self.scorer_name is not None, "Need to start callback with the model name."

        if iteration_id % self.freq == 0:
            torch.save({
                'config': model.get_config(),
                'model_state_dict': model.state_dict(),
            }, self.full_checkpoint_dir / "best_checkpoint.pt")

        if kwargs["eval_loss"] < self.best_loss:
            logger.info(f"Saving checkpoint for epoch {iteration_id}.")

            self.best_loss = kwargs["eval_loss"]
            torch.save({
                'config': model.get_config(),
                'model_state_dict': model.state_dict(),
            }, self.full_checkpoint_dir /  "best_checkpoint.pt")
     