import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error (RMSE) loss function.

    This loss function calculates the RMSE between predicted and target values.

    Args:
        eps (float, optional): Small value to avoid division by zero in the square root. Default is 1e-6.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        """
        Forward pass method to compute the RMSE loss.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Computed RMSE loss.
        """
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss
    

class ConstrainedLoss(nn.Module):
    def __init__(self, base_loss, lower_bound=0, upper_bound=100, penalty_weight=1):
        super().__init__()
        self.base_loss = base_loss
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.penalty_weight = penalty_weight


    def forward(self, predictions, targets):
        loss = self.base_loss(predictions, targets)

        penalty_lower = torch.sum(torch.relu(self.lower_bound - predictions))
        penalty_upper = torch.sum(torch.relu(predictions - self.upper_bound))

        total_loss = loss + self.penalty_weight * (penalty_lower + penalty_upper)
        return total_loss
