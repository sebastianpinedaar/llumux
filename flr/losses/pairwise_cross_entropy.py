import torch

class PairwiseCrossEntropyLoss:
    def __init__(self):
        pass

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
        y_ij = y_ij.reshape(-1)
        y_hat1 = torch.nn.Sigmoid()(scores_i - scores_j).reshape(-1)    
        y_hat2 = torch.nn.Sigmoid()(scores_j - scores_i).reshape(-1)
        y2 = (y_ij == -1).int()
        y1 = (y_ij == 1).int()
        loss1 = -y1 * torch.log(y_hat1) - (1 - y1) * torch.log(1-y_hat1)
        loss2 = -y2 * torch.log(y_hat2) - (1 - y2) * torch.log(1-y_hat2)
        loss = loss1 + loss2
        return loss.mean()  #
