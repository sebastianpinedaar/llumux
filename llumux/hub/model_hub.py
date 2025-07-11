import yaml
import os

from pathlib import Path

class ModelHub:
    """Class to manage model hub configurations and model cards."""
    def __init__(self, model_hub_name: str = "llm_instruct"):
        self.config_file = Path(os.environ["FLR_HOME"]) / "config" / "model_hubs" / f"{model_hub_name}.yml"

        with open(self.config_file, "r") as f:
            self.model_info = yaml.safe_load(f)
        
        self.model_cards = self.model_info["models"]
        
    def get_models(self):
        return list(self.model_cards.keys())
    
    def append_model(self, model_name, model_card):
        self.models[model_name] = model_card

    def get_attributes_from_model_card(self, attribute_name):
        return [model_card[attribute_name] for model_card in self.model_cards.values()]
    
    def get_simplified_model_cards(self, attribute_name):
        """
        Returns a list of dictionaries with the specified attribute from each model card.
        """
        simplified_model_cards = {}
        for model_name, model_card in self.model_cards.items():
            if attribute_name in model_card:
                simplified_model_cards[model_name] = model_card[attribute_name]
            else:
                simplified_model_cards[model_name] = None
        return simplified_model_cards