import torch

from ..tile_level.regression import BaseRegressionEvaluator


class PLRegressionEvaluator(BaseRegressionEvaluator):
    """Special class for evaluation PL Regression models that give the output a 2D matrix of numbers."""

    MetricsPool = frozenset([
        'r2_score',
        'MAE',
        'MSE',
        'RMSE',
        'MAPE'
    ])

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Updates the state of the buffer by accumulating given predictions and labels."""

        if y_true.size() != y_pred.size():
            raise ValueError(
                f"True labels shape {y_true.shape} must have match with pred {y_pred.shape} shape, got: {y_true.shape}"
            )
        if len(y_pred.size()) != 4:
            raise ValueError(f"Prediction must have shape B X C X H X W, got={y_pred.shape}")

        self._update_buffer(y_pred, y_true)
