import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn

from ..base import Loss


class L1Loss(nn.L1Loss, Loss):
    """Standard L1 loss."""
    pass


class L2Loss(nn.MSELoss, Loss):
    """Standard L2 or MSE loss."""
    pass


class SmoothL1Loss(nn.SmoothL1Loss, Loss):
    """See: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss."""
    pass


class HuberLoss(nn.HuberLoss, Loss):
    """See: https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss"""
    pass


class PinBallLoss(Loss):
    """
    The main loss for Quantile regression.

        source - https://github.com/ywatanabe1989/custom_losses_pytorch
        The formula for pinball loss is
        L_α(d,f) = (d-f) α    if d≥f
                   (f-d)(1-α) if f>d
        where α is the quantile, f is the quantile forecast, d is the demand
    """

    def __init__(self, quantiles: List = (0.1, 0.5, 0.9)):
        super().__init__(name="PinBallLoss")
        self.quantiles = quantiles

    @staticmethod
    def _calculate_for_single_q(y_pred: torch.Tensor, y_true: torch.Tensor, i: int, q: float):
        """
        Calculates the single quantile loss.
        Args:
            y_pred: net output with shape [B x NClasses]
            y_true: ground truth with shape [B]

        Returns:
            the single quantile loss
        """
        errors = y_true[:, 0] - y_pred[:, i]
        return torch.max((q - 1) * errors, q * errors).unsqueeze(1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Args:
            y_pred: net output with shape [B x N quantiles]
            y_true: ground truth with shape [B x 1]

        Returns: the scalar loss value averaged over the batch
        """
        batch_size_pred, n_quantiles = y_pred.shape[:2]
        batch_size_actual, n_target = y_true.shape[:2]
        assert batch_size_pred == batch_size_actual, f"Target={batch_size_actual}, Pred={batch_size_pred}"
        assert n_quantiles == len(self.quantiles), f"Pred={n_quantiles}, GivenQuantiles={self.quantiles}"
        assert n_target == 1, f"Target={n_target} must be 1"

        losses = [self._calculate_for_single_q(y_pred, y_true, i, q) for i, q in enumerate(self.quantiles)]
        losses = torch.cat(losses, dim=1)  # each row represents the losses for each sample
        return torch.mean(torch.sum(losses, dim=1))


class MDNLoss(Loss):
    """Mixture Density Loss for MDN regression."""
    EPSILON = 1e-15

    ONE_DIV_SQRT_2PI = 1.0 / math.sqrt(2 * math.pi)
    LOG_2PI = math.log(2 * math.pi)

    @classmethod
    def gaussian_probability(cls, target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        """Calculates gaussian cdf """
        sigma = sigma + cls.EPSILON
        exponent_argument = torch.pow((target.expand_as(mu) - mu) / sigma, 2)
        return (cls.ONE_DIV_SQRT_2PI / sigma) * torch.exp(-0.5 * exponent_argument)

    @classmethod
    def log_gaussian_probability(cls, target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        """Calculates gaussian cdf """
        sigma = sigma + cls.EPSILON
        return -0.5 * cls.LOG_2PI - torch.log(sigma) - 0.5 * torch.pow((target.expand_as(mu) - mu) / sigma, 2)

    def forward(self,
                y_pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                y_true: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Args:
            y_pred: net output that is tuple - pi, mu, sigma,
                    each of them must be a tensor with shape [B, NComponents, ...]
            y_true: ground truth with shape [B, ...]
            mask: optional tensor that indicates which pixels should be taken into consideration

        Returns:
            negative log likelihood
        """
        pi, mu, sigma = y_pred
        prob = self.gaussian_probability(y_true, mu, sigma)
        nll = -torch.log(torch.sum(prob * pi, dim=1) + self.EPSILON)  # summing across components
        if mask is not None:
            nll = torch.masked_select(nll, mask.squeeze() > 0.5)  # selecting only pixels that are in mask
        nll = torch.mean(nll)
        return nll
