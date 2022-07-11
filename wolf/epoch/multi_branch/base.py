"""The module implements Multi-Branch (aka model has multiple branches or heads) Train and Validation epochs."""
from collections import defaultdict
from typing import Set, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from yield_forecasting.utils.torch_helpers import to_device
from ..single_branch.base import SingleBranchTrainEpoch, SingleBranchValidationEpoch, SingleBranchEpochMixin
from ...losses import MultiBranchLoss
from ...metrics import AverageValueMeter, MultiBranchEvaluator


class MultiBranchEpochMixin(SingleBranchEpochMixin):
    """Defines the mixin class multi branch epochs."""

    # internals
    _network: nn.Module
    _loss: MultiBranchLoss
    _evaluator: MultiBranchEvaluator
    _results_keeper: Dict
    _device: str or torch.device

    @property
    def loss(self) -> MultiBranchLoss:
        return self._loss

    @property
    def evaluator(self) -> MultiBranchEvaluator:
        return self._evaluator

    def get_target_prediction(self, data_sample: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        target = to_device(data_sample['label'], device=self._device, detach=False)
        return target, self._network.forward(data_sample['input_image'].to(self._device))

    def get_results_keeper(self) -> Dict:
        return {
            'epoch_metrics': {},
            'loss': AverageValueMeter(),
            'branches_losses': {
                f'{branch}/{loss.name}': AverageValueMeter()
                for branch, loss in self._loss.branches_losses.items()
            },
            'images': defaultdict(dict)
        }

    def get_epoch_metrics(self) -> Dict:
        """Returns epoch-wise calculated metrics and loss mean and std."""
        metrics = {
            f'{branch}/{metric}': metric_value
            for branch, branch_metrics in self.results_keeper['epoch_metrics'].items()
            for metric, metric_value in branch_metrics.items()
        }
        branches_losses = {
            branch_loss_name: float(loss_keeper.mean)
            for branch_loss_name, loss_keeper in self.results_keeper['branches_losses'].items()
        }
        loss_keeper = self.results_keeper['loss']
        return {
            self._loss.name: float(loss_keeper.mean),
            f'{self._loss.name}_std': float(loss_keeper.std),
            **branches_losses,
            **metrics,
        }

    def register_loss(self, loss: torch.Tensor, branches_losses: Dict = None) -> None:
        """Registers metrics and loss."""
        self._results_keeper['loss'].add(loss.numpy())

        if branches_losses:
            for branch, branch_loss_value in branches_losses.items():
                self._results_keeper['branches_losses'][branch].add(branch_loss_value.numpy())

    def register_metrics(self, y_true: Dict, y_pred: Dict, **kwargs) -> None:
        """Registers metrics and loss."""
        self._evaluator.update(y_pred=y_pred, y_true=y_true)


class MultiBranchTrainEpoch(MultiBranchEpochMixin, SingleBranchTrainEpoch):
    """Base MultiBranch Train Epoch."""

    def dataloader_loop(self, dataloader: DataLoader, save_indices: Set[int] = None, valid_metric: float = None):
        """Performs loop over dataloader."""
        description = f'    {self._stage_name.upper()[:5]} BATCHES'
        for i, data_sample in enumerate(tqdm(dataloader, desc=description, total=len(dataloader))):
            loss, y_true, y_pred = self.batch_update(data_sample)

            loss = to_device(loss, device='cpu', detach=False)
            branches_losses = to_device(self.loss.last_computed_state, device='cpu', detach=True)
            self.register_loss(loss, branches_losses=branches_losses)

            self.register_metrics(y_true, y_pred)  # metrics can be calculated on GPU
            (y_true, y_pred) = to_device((y_true, y_pred), device='cpu', detach=False)
            self.register_sample(i, data_sample.get('debug_band'), y_true, y_pred, saving_batches=save_indices)
            self.update_lr(is_batch=True, metric_value=valid_metric)


class MultiBranchValidationEpoch(MultiBranchEpochMixin, SingleBranchValidationEpoch):
    """Base MultiBranch Validation Epoch."""

    def dataloader_loop(self, dataloader: DataLoader, save_indices: Set[int] = None, valid_metric: float = None):
        """Performs loop over dataloader."""
        description = f'    {self._stage_name.upper()[:5]} BATCHES'
        for i, data_sample in enumerate(tqdm(dataloader, desc=description, total=len(dataloader))):
            loss, y_true, y_pred = self.batch_update(data_sample)

            loss = to_device(loss, device='cpu', detach=False)
            branches_losses = to_device(self.loss.last_computed_state, device='cpu', detach=True)
            self.register_loss(loss, branches_losses=branches_losses)

            self.register_metrics(y_true, y_pred)  # metrics can be calculated on GPU
            (y_true, y_pred) = to_device((y_true, y_pred), device='cpu', detach=False)
            self.register_sample(i, data_sample.get('debug_band'), y_true, y_pred, saving_batches=save_indices)
