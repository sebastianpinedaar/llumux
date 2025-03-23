import torch

class PairwiseLogisticLoss:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def __call__(self, scores, y_ij):
        """
        Computes the pairwise logistic loss.

        Args:
            scores_i (torch.Tensor): Scores assigned by the model to item i.
            scores_j (torch.Tensor): Scores assigned by the model to item j.
            y_ij (torch.Tensor): Pairwise labels (1 if i should be ranked higher than j, -1 otherwise).
        
        Returns:
            torch.Tensor: The computed logistic loss.
        """
        assert len(scores) == 2
        scores_i, scores_j = scores
        margin = scores_i - scores_j  # Difference in scores
        loss = torch.log(1 + torch.exp(-(y_ij+self.epsilon) * margin.reshape(-1)))  # Logistic loss
        return loss.mean()  #
