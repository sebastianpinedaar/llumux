import os

import torch
import logging
import numpy as np

from pathlib import Path

from .base_callback import BaseCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointSaver(BaseCallback):
    """Callback to save checkpoints during training."""
    def __init__(self, 
                  workspace_path: str = None,
                  freq: int = 1,
                  checkpoint_dir:str ="scorers"):
        """Callback to save checkpoints during training.
        Args:
            workspace_path (str): Path to the workspace directory.
            freq (int): Frequency of saving checkpoints.
            checkpoint_dir (str): Directory to save checkpoints.
        """
        if workspace_path is None:
            workspace_path = Path(os.environ.get("FLR_HOME", "")) / "workspace"
        self.workspace_path = Path(workspace_path) 
        self.freq = freq
        self.checkpoint_dir = checkpoint_dir

    def start(self, scorer_name: str):
        """Initialize the callback with the scorer name and create the checkpoint directory.
        Args:
            scorer_name (str): Name of the scorer for which checkpoints are saved.
        """
        self.scorer_name = scorer_name
        self.full_checkpoint_dir = self.workspace_path / self.checkpoint_dir / self.scorer_name
        self.full_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = np.inf

    def on_batch_end(self, iteration_id, model, **kwargs):
        """Save the model checkpoint if the frequency condition is met and if the loss is better than the best loss.
        Args:
            iteration_id (int): Current iteration ID.
            model (torch.nn.Module): The model to save.
            **kwargs: Additional keyword arguments, including eval_loss.
        """
        
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
     