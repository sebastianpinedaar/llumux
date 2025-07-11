from ..hub.model_hub import ModelHub

LAST_HIDDEN_DIM = ModelHub("prompt_embedders").get_simplified_model_cards("last_hidden_dim")