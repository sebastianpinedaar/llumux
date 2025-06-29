import torch

class PairwiseCrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        """
        Computes the pairwise logistic loss.

        Args:
            y_pred_i (torch.Tensor): y_pred assigned by the model to item i.
            y_pred_j (torch.Tensor): y_pred assigned by the model to item j.
            y_true (torch.Tensor): Pairwise labels (1 if i should be ranked higher than j, -1 otherwise).
        
        Returns:
            torch.Tensor: The computed logistic loss.
        """

        if isinstance(y_pred, list):
            assert len(y_pred) == 2
            y_pred_i, y_pred_j = y_pred
            y_true = y_true.reshape(-1)
        else:
            assert y_pred.shape[1] == 2, "y_pred should have two columns for pairwise comparison"
            assert y_pred.shape == y_true.shape
            y_pred_i, y_pred_j = y_pred[:, 0], y_pred[:, 1]
            y_true = torch.sign(y_true[:,1]-y_true[:, 0])
        y_hat1 = torch.nn.Sigmoid()(y_pred_i - y_pred_j).reshape(-1)    
        y_hat2 = torch.nn.Sigmoid()(y_pred_j - y_pred_i).reshape(-1)
        y2 = (y_true == -1).int()
        y1 = (y_true == 1).int()
        loss1 = -y1 * torch.log(y_hat1) - (1 - y1) * torch.log(1-y_hat1)
        loss2 = -y2 * torch.log(y_hat2) - (1 - y2) * torch.log(1-y_hat2)
        loss = loss1 + loss2
        return loss.mean()  #


