import torch.nn as nn

from .pairwise_logistic_loss import PairwiseLogisticLoss
from .pairwise_cross_entropy import PairwiseCrossEntropyLoss
from .list_mle import ListMLELoss

loss_functions_map = {
    "pairwise_logistic_loss": PairwiseLogisticLoss,
    "pairwise_cross_entropy": PairwiseCrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
    "list_mle": ListMLELoss
}