from .pairwise_logistic_loss import PairwiseLogisticLoss
from .pairwise_cross_entropy import PairwiseCrossEntropyLoss

LOSS_FUNCTIONS = {
    "pairwise_logistic_loss": PairwiseLogisticLoss,
    "pairwise_cross_entropy": PairwiseCrossEntropyLoss
}