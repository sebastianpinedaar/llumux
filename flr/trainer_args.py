class TrainerArgs:

    def __init__(self, batch_size: int,
                     epochs: int, 
                     lr: float, 
                     seed: int, 
                     device: str,
                     **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = device
        self.kwargs = kwargs