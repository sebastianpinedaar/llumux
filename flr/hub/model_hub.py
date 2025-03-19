from .model_card import ModelCard

class ModelHub:
    def __init__(self, model_cards: list[ModelCard]):
        self.models = model_cards

    def get_models(self):
        return self.models
    
    def append_model(self, model_card: ModelCard):
        self.models.append(model_card)