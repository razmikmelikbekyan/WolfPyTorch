import torch

from ..base import TemporalLoss
from ...single_branch.image_level.classification import CrossEntropyLoss, MultiClassFocalLoss

__all__ = [
    'TemporalCrossEntropyLoss',
    'TemporalMultiClassFocalLoss',
]


class TemporalCrossEntropyLoss(CrossEntropyLoss, TemporalLoss):
    """The temporal version of the CrossEntropyLoss."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Args:
            y_pred: net output with shape [B x T x NCLasses]
            y_true: ground truth with shape [B x T]

        Returns: the scalar loss value averaged over the batch
        """
        B, T, N = y_pred.shape
        return sum(super(TemporalCrossEntropyLoss, self).forward(y_pred[:, i], y_true[:, i]) for i in range(T))


class TemporalMultiClassFocalLoss(MultiClassFocalLoss, TemporalLoss):
    """The temporal version of the MultiClassFocalLoss"""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Args:
            y_pred: net output with shape [T x B x NCLasses]
            y_true: ground truth with shape [T x B]

        Returns: the scalar loss value averaged over the batch
        """
        B, T, N = y_pred.shape
        return sum(super(TemporalMultiClassFocalLoss, self).forward(y_pred[:, i], y_true[:, i]) for i in range(T))
