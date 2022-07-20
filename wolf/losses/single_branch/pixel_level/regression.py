from typing import Dict

import torch

from .._commons.regression import L1Loss, L2Loss, SmoothL1Loss, HuberLoss, PinBallLoss, MDNLoss
from ..base import Loss

__all__ = [
    'L1Loss',
    'SmoothL1Loss',
    'L2Loss',
    'HuberLoss',
    'PinBallLoss',
    'MDNLoss',
    'L2PinballLoss'
]


class L2PinballLoss(Loss):
    """L2Loss + Pinball Loss"""

    def __init__(self, mse_kwargs: Dict, pinball_kwargs: Dict):
        super().__init__(name="MSEPinballLoss")
        self.mse_loss = L2Loss(**mse_kwargs)
        self.pinball_loss = PinBallLoss(**pinball_kwargs)

        quantiles = pinball_kwargs['quantiles']
        assert isinstance(quantiles, list)
        self.middle_quantile_index = len(quantiles) // 2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pinball_loss = self.pinball_loss(y_pred.mean(dim=(2, 3)), y_true.mean(dim=(2, 3)))
        mse_loss = self.mse_loss(y_pred[:, self.middle_quantile_index:self.middle_quantile_index + 1, ...], y_true)
        return mse_loss + pinball_loss
