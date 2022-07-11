from typing import Dict

from .regularization import WeightsRegularization


def get_regularizer(name: str, **regularization_loss_kwargs: Dict) -> WeightsRegularization:
    """Returns the regularization loss."""
    return WeightsRegularization(name, **regularization_loss_kwargs)
