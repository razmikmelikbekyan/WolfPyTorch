import torch

from ..tile_level.classification import BaseClassificationEvaluator


class PLClassificationEvaluator(BaseClassificationEvaluator):
    """Special class for PL Classification models."""

    ConfusionMatrixMetrics = frozenset([
        'accuracy',
        'balanced_accuracy',
        'confusion_matrix',
        'recall',
        'precision',
        'f1',
        'iou'
    ])
    ScoreRequiringMetrics = frozenset([])

    MetricsPool = ConfusionMatrixMetrics

    def _check_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Checks the input shapes.

        Predictions and targets are supposed to fall into one of these categories (and are
        validated to make sure this is the case):

        * Both y_pred and y_true are of shape ``(B, ...)``, and both are integers (multi-class)
        * Both y_pred and y_true are of shape ``(B, 1)``, and y_true is binary, while y_pred
          are a float (binary) (already flattened)
        * Both y_pred and y_true are of shape ``(B, 1, H, W)``, and y_true is binary, while y_pred
          are a float (binary)
        * y_pred are of shape ``(B, C, H, W)`` and are floats, and y_true is of shape ``(B, 1, H, W)`` and
          is integer (multi-class)
        """
        if self._n_classes == 2:
            if y_true.dim() <= 2 and y_pred.dim() <= 2 and y_true.shape == y_pred.shape:
                return

        if self._prediction_type == 'label':
            if y_pred.size() != y_true.size():
                raise ValueError(f'y_pred and y_true must have the same size, got {y_pred.size()} and {y_true.size()}')
        else:
            if y_true.dim() != 4 or y_true.shape[1] != 1:
                raise ValueError(f'y_true must have 4 dimensions (B x 1 x H x W), got {y_true.shape}')

            if y_pred.dim() != 4:
                raise ValueError(f'y_pred must have 4 dimensions (B x n_classes x H x W), got {y_pred.shape}')

            if self._n_classes == 2:
                if y_pred.shape[1] != 1:
                    raise ValueError(f'In case of binary prediction y_pred must have 4 dimensions (B x 1 x H x W), '
                                     f'got {y_pred.shape}')
            else:
                if y_pred.shape[1] != self._n_classes:
                    raise ValueError(f'In case of multiclass prediction y_pred must have 4 dimensions '
                                     f'(B x n_classes x H x W), got {y_pred.shape}')
