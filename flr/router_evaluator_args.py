import torch

class RouterEvaluatorArgs:
    """
    Class to hold the arguments for the evaluator.
    """

    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = kwargs.get("batch_size", 32)

