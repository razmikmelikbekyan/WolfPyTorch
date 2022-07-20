from typing import Dict, Optional, Union

import torch
from ..single_branch.base import Loss


class MultiBranchLoss(Loss):
    """The special loss for MULTI BRANCH architectures."""

    def __init__(self, branches_losses: Dict[str, Loss], branches_weights: Union[Optional[Dict], Optional[str]] = None):
        """
        Args:
            branches_losses: the unique branch name and its respective loss object
            branches_weights: the unique branch name and its respective loss weight
        """
        if branches_weights:
            assert set(branches_losses.keys()) == set(branches_weights.keys())
        else:
            branches_weights = {task: 1. / len(branches_losses) for task in branches_losses.keys()}

        super().__init__(name="+".join(f'{loss.name}' for branch, loss in branches_losses.items()))
        self.branches_losses = branches_losses
        self.branches_weights = branches_weights

        self._last_computed_state = {}

    @property
    def last_computed_state(self) -> Optional[Dict[str, float]]:
        """Returns the last computed loss value for each branch."""
        return self._last_computed_state

    def __call__(self,
                 predictions: Dict[str, torch.Tensor],
                 targets: Dict[str, torch.Tensor],
                 **kwargs: Dict[str, Dict]) -> float:
        """
        Args:
            targets: the targets for each branch
            predictions: the predictions for each branch
            kwargs: additional parameter to pass loss function

        Returns:
            computes the loss for each branch and returns the weighted sum of them
        """
        branch_loss_values = {
            branch: loss(predictions[branch], targets[branch], **kwargs.get(branch, {}))
            for branch, loss in self.branches_losses.items()
        }
        total_loss_value = sum(
            self.branches_weights[branch] * branch_loss_values[branch] for branch in self.branches_losses.keys()
        )

        self._last_computed_state = {
            f'{branch}/{self.branches_losses[branch].name}': branch_loss_value
            for branch, branch_loss_value in branch_loss_values.items()
        }
        return total_loss_value

    def to(self, *args, **kwargs):
        for loss in self.branches_losses.values():
            loss.to(*args, **kwargs)
