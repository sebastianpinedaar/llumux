class TrainerArgs:

    def __init__(self, batch_size: int = 32,
                     epochs: int = 5, 
                     lr: float = 0.01, 
                     seed: int = 42, 
                     device: str = "cuda",
                     checkpoint_name: str = "checkpoint.pt",
                     checkpoint_dir: str = "checkpoints",
                     eval_freq: int = 1,
                     **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = device
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = checkpoint_dir
        self.eval_freq = eval_freq
        self.kwargs = kwargs