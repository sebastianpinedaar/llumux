from .checkpointing import CheckpointSaver
from .loss_tracker import LossTracker
from ..hub.model_hub import ModelHub

LAST_HIDDEN_DIM = ModelHub("prompt_embedders").get_attributes_from_model_card("last_hidden_dim")