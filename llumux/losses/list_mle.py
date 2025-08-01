
import torch

class ListMLELoss(torch.nn.Module):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    This class is a wrapper for the listMLE function.
    """
    
    def __init__(self, eps=1e-6, padded_value_indicator=-1):
        """
        Initializes the ListMLELoss class.
        Args:
        - eps: epsilon value, used for numerical stability
        - padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        """
        super(ListMLELoss, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator

    def forward(self, y_pred, y_true):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        Args:
        - y_pred: predictions from the model, shape [batch_size, slate_length]
        - param y_true: ground truth labels, shape [batch_size, slate_length]
        
        Returns:
        - loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == self.padded_value_indicator

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))

