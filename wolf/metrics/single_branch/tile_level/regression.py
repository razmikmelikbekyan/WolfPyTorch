from copy import deepcopy
from typing import List, Dict, Tuple, Union

import torch
from torchmetrics.functional import mean_absolute_error, mean_squared_error, r2_score
from torchmetrics.functional.regression.mape import _mean_absolute_percentage_error_update

from ...base import BaseEvaluator


class BaseRegressionEvaluator(BaseEvaluator):
    """A Base class for all regression evaluators."""

    MetricsPool: frozenset

    def __init__(self, metrics: List[Tuple[str, Dict]], epsilon: float = None,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            metrics: the names of metrics and respective kwargs for calculating them
            epsilon: the value that is used to filter out small values
        """
        super(BaseRegressionEvaluator, self).__init__(metrics=metrics)
        self._epsilon = epsilon
        self._history_is_required = False
        self.register_buffer('n_samples', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('absolute_error', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('squared_error', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('absolute_percentage_error', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('true_sum', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('true_squared_sum', torch.tensor(0., dtype=torch.float32))

        self._state = {}

        self.to(device=device)

    @staticmethod
    def get_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return r2_score(target=y_true, preds=y_pred)

    @staticmethod
    def get_MAE(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return mean_absolute_error(target=y_true, preds=y_pred)

    @staticmethod
    def get_MSE(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return mean_squared_error(target=y_true, preds=y_pred)

    @staticmethod
    def get_RMSE(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return mean_squared_error(target=y_true, preds=y_pred, squared=False)

    @staticmethod
    def get_MAPE(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(y_pred, y_true)
        return sum_abs_per_error / num_obs

    def _average(self, input_tensor: Union[torch.Tensor, int]) -> torch.Tensor:
        """A shorthand for normalizing values by n_samples."""
        return input_tensor / self.n_samples

    def _update_buffer(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Updates a buffer."""
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)

        # applying mask before calculating the metrics
        if self._epsilon is not None:
            mask = y_true > self._epsilon
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        n_samples = y_true.numel()
        self.n_samples += n_samples
        self.true_sum += torch.sum(y_true)
        self.true_squared_sum += torch.sum(y_true * y_true)

        self.squared_error += self.get_MSE(y_true=y_true, y_pred=y_pred) * n_samples
        self.absolute_error += self.get_MAE(y_true=y_true, y_pred=y_pred) * n_samples
        self.absolute_percentage_error += self.get_MAPE(y_true=y_true, y_pred=y_pred) * n_samples

    def compute(self) -> Dict:
        for metric_name, metric_kwargs in self._metrics:
            if metric_kwargs:
                saving_name = f'{metric_name}-{"-".join(f"{k}={v}" for k, v in metric_kwargs.items())}'
            else:
                saving_name = metric_name

            if metric_name == 'MSE':
                metric_value = self._average(self.squared_error)
            elif metric_name == 'RMSE':
                metric_value = self._average(self.squared_error) ** 0.5

            elif metric_name == 'MAE':
                metric_value = self._average(self.absolute_error)
            elif metric_name == 'MAPE':
                metric_value = self._average(self.absolute_percentage_error)

            elif metric_name == 'r2_score':
                total_sum_squared = self.true_squared_sum - self._average(self.true_sum ** 2)
                metric_value = 1 - (self.squared_error / total_sum_squared)

            else:
                raise NotImplementedError(f"For given metric_name={metric_name} there is no implementation.")

            self._state[saving_name] = metric_value
        return self.state

    def reset(self):
        self.n_samples.zero_()
        self.absolute_error.zero_()
        self.squared_error.zero_()
        self.absolute_percentage_error.zero_()
        self.true_sum.zero_()
        self.true_squared_sum.zero_()

    @property
    def state(self) -> Dict:
        return self.to_python_float(deepcopy(self._state))


class TLRegressionEvaluator(BaseRegressionEvaluator):
    """Special class for evaluation tile level Regression models that give the output a single number."""

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
        if len(y_true.shape) != 2:
            raise ValueError(f"True labels must have (batch size, 1) shape, got: {y_true.shape}")

        if y_true.shape[1] != 1:
            raise ValueError(f"True labels must have (batch size, 1) shape, got: {y_true.shape}")

        self._update_buffer(y_pred, y_true)
