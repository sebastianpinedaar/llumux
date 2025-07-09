from .general_scorer import GeneralScorer

class ListwiseScorer(GeneralScorer):
    def __init__(self, *args, **kwargs):
        super(ListwiseScorer, self).__init__(*args, **kwargs)