from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Any, Optional

import numpy as np
import torch

__all__ = ['BaseEvaluator']


class BaseEvaluator(ABC, torch.nn.Module):
    """Base Class for all evaluators."""

    MetricsPool: frozenset

    def __init__(self, metrics: List[Tuple[str, Dict]]):
        """
        Args:
            metrics: the names of metrics and respective kwargs for calculating them
        """
        super(BaseEvaluator, self).__init__()
        for metric_name, _ in metrics:
            if metric_name not in self.MetricsPool:
                raise ValueError(f"Given {metric_name} metric is wrong, please select from {list(self.MetricsPool)}.")

        self._metrics = metrics
        self._evaluation_buffer_keys = tuple()

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Overwriting register_buffer method to keep evaluation related buffer keys only."""
        super().register_buffer(name, tensor=tensor, persistent=persistent)
        self._evaluation_buffer_keys += (name,)

    @property
    def evaluation_buffer(self) -> Dict[str, torch.Tensor]:
        """Returns the evaluation buffer at current state."""
        return {k: self._buffers[k] for k in self._evaluation_buffer_keys}

    def update_evaluation_buffer(self, buffer_update: Dict[str, torch.Tensor]) -> None:
        """Updates the evaluation buffer buy adding the buffer update to existing buffer."""
        for k in self._evaluation_buffer_keys:
            try:
                update_value = buffer_update[k]
            except KeyError:
                raise KeyError(f"buffer_update must contain all evaluation buffer keys, '{k}' is missing.")

            if not isinstance(update_value, torch.Tensor):
                raise TypeError(f"update buffer value for {k} must be tensor, got: {type(update_value)}")

            if str(self._buffers[k].dtype) != str(update_value.dtype):
                raise ValueError(f"Update value for for {k} expected to "
                                 f"have {self._buffers[k].dtype} dtype, got: {update_value.dtype}")

            self._buffers[k] += update_value

    @staticmethod
    def tensor_to_float(input_tensor: Union[torch.Tensor, Any]) -> Union[np.ndarray, float, Any]:
        """Convert tensor to float."""
        if not isinstance(input_tensor, torch.Tensor):
            return input_tensor
        elif input_tensor.numel() == 1:
            return input_tensor.cpu().item()
        else:
            return input_tensor.cpu().numpy()

    @classmethod
    def to_python_float(cls, state_dict: Dict) -> Dict[str, Union[np.ndarray, float]]:
        """Converts a collection of metrics to python float."""
        for metric_name, metric_value in state_dict.items():
            if isinstance(metric_value, dict):
                cls.to_python_float(metric_value)
            else:
                state_dict.update({metric_name: cls.tensor_to_float(metric_value)})
        return state_dict

    @abstractmethod
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Override this method to update the state variables of your evaluator class."""
        pass

    @abstractmethod
    def compute(self) -> Dict:
        """Override this method to compute the final metrics values from state variables."""
        pass

    @abstractmethod
    def reset(self):
        """Override this method to reset the evaluator for starting new epoch."""
        pass
