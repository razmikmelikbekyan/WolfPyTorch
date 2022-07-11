from typing import Dict

import torch
from intelinair_ml.evaluators import BaseEvaluator


class MultiBranchEvaluator(BaseEvaluator):
    """The special evaluator that combines multiple evaluators."""

    def __init__(self, branches_evaluators: Dict[str, BaseEvaluator]):
        """
        Args:
            branches_evaluators: the unique branch name and its respective evaluator object
        """
        self.branches_evaluators = branches_evaluators

    def update(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> None:
        """
        Updates each evaluator.
        Args:
            y_true: the targets for each branch
            y_pred: the predictions for each branch
        """
        for branch, evaluator in self.branches_evaluators.items():
            evaluator.update(y_pred[branch], y_true[branch])

    def compute(self) -> Dict[str, Dict]:
        """
        Computes each evaluator.
        Returns:
            branch name - branch metrics dict
        """
        return {branch: evaluator.compute() for branch, evaluator in self.branches_evaluators.items()}

    def reset(self) -> None:
        """Resets each evaluator."""
        for evaluator in self.branches_evaluators.values():
            evaluator.reset()
