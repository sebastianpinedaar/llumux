import torch

class PairwiseLogisticLoss:
    """Pairwise Logistic Loss for pairwise ranking tasks."""
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

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
        if isinstance(y_pred, list) or isinstance(y_pred, tuple):
            assert len(y_pred) == 2
            y_pred_i, y_pred_j = y_pred
            y_true = y_true.reshape(-1)
        else:
            assert y_pred.shape[1] == 2, "y_pred should have two columns for pairwise comparison"
            assert y_pred.shape == y_true.shape
            y_pred_i, y_pred_j = y_pred[:, 0], y_pred[:, 1]
            y_true = torch.sign(y_true[:,1]-y_true[:, 0])
 
        margin = (y_pred_i - y_pred_j).reshape(-1)  # Difference in y_pred
        loss = torch.log(1 + torch.exp(-y_true[y_true!=0] * margin[y_true!=0]))
        loss2 = self.epsilon * torch.abs(margin[y_true==0])  # Logistic loss
        if loss2.shape[0] == 0:
            loss2 = torch.tensor(0.0).to(loss.device)
        return loss.mean() + loss2.mean()  #
