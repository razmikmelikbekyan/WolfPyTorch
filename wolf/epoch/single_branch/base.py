"""The module provides base mixins training and validation epochs"""
import gc
from collections import defaultdict
from typing import Dict, Tuple, Set, Any

import torch
from pytorch_warmup import BaseWarmup
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from wolf.utils.torch_helpers import to_device, detach_object
from ...losses import Loss
from ...metrics import BaseEvaluator, AverageValueMeter
from ...optimization import EPOCH_UPDATE_SCHEDULERS, METRIC_BASED_SCHEDULERS
from ...regularization import WeightsRegularization


class SingleBranchEpochMixin:
    """Defines the mixin class epochs."""

    # internals
    _network: nn.Module
    _loss: Loss
    _evaluator: BaseEvaluator
    _results_keeper: Dict
    _device: str or torch.device

    @property
    def network(self) -> nn.Module:
        return self._network

    @property
    def loss(self) -> Loss:
        return self._loss

    @property
    def loss_name(self) -> str:
        return self._loss.name

    @property
    def evaluator(self) -> BaseEvaluator:
        return self._evaluator

    @property
    def results_keeper(self) -> Dict:
        return self._results_keeper

    def _to_device(self):
        """Sets device."""
        self._network.to(self._device)
        self._loss.to(self._device)

    @staticmethod
    def get_results_keeper() -> Dict:
        return {
            'epoch_metrics': {},
            'loss': AverageValueMeter(),
            'images': defaultdict(dict)
        }

    def get_epoch_metrics(self) -> Dict:
        """Returns epoch-wise calculated metrics and loss mean and std."""
        loss_keeper = self.results_keeper['loss']
        return {
            self._loss.name: float(loss_keeper.mean),
            f'{self._loss.name}_std': float(loss_keeper.std),
            **self.results_keeper['epoch_metrics'],
        }

    def get_epoch_images(self) -> Dict:
        """Returns debug images."""
        return self.results_keeper['images']

    def get_target_prediction(self, data_sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the target and prediction from data sample.
        It will be further used for loss calculation and optimizer.
        This method is separated from the other staff in order to let developer to modify it in subclasses in case
        there is a need to modify prediction before loss calculation.

        Args:
            data_sample: the input data sample
        Returns:
            target: the target tensor
            prediction: the prediction of model
        """
        return data_sample['label'].to(self._device), self._network.forward(data_sample['input_image'].to(self._device))

    def register_loss(self, loss: torch.Tensor, **kwargs: Any) -> None:
        """Registers loss."""
        self._results_keeper['loss'].add(loss.numpy())

    def register_metrics(self, y_true: Any, y_pred: Any, **kwargs: Any) -> None:
        """Registers metrics."""
        self._evaluator.update(y_pred=y_pred, y_true=y_true)

    def register_sample(self,
                        batch_number: int,
                        debug_image: torch.Tensor,
                        y_true: Any,
                        y_pred: Any,
                        saving_batches: Set[int] = None,
                        **kwargs: Any) -> None:
        """Registers samples prediction in the images keeper."""
        raise NotImplementedError("Each Epoch should implement own registration method since the inputs may"
                                  " vary from task to task.")


class SingleBranchTrainEpoch(SingleBranchEpochMixin):
    """Single Branch Train Epoch."""

    _stage_name = "train"

    def __init__(self,
                 network: nn.Module,
                 loss: Loss,
                 evaluator: BaseEvaluator,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 warmup_scheduler: BaseWarmup = None,
                 regularization: WeightsRegularization = None,
                 device: str = 'cpu',
                 verbose: bool = True,
                 ):
        """
        Args:
            network: the PyTorch model
            loss: the Loss object, like CrossEntropy
            evaluator: the evaluator object, like Classification Evaluator
            optimizer: the optimizer object, like Adam
            lr_scheduler: the learning rate scheduler object
            warmup_scheduler: the warmup scheduler for learning rate
            regularization: the regularization of the network weights
            device: cpu or cuda
            verbose: if True will show intermediate results
        """
        self._network = network
        self._loss = loss
        self._evaluator = evaluator
        self._verbose = verbose
        self._device = device
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_update_frequency = self._get_lr_update_frequency(lr_scheduler)
        self._warmup_scheduler = warmup_scheduler
        self._regularization = regularization

        self._results_keeper = self.get_results_keeper()
        self._to_device()

    @property
    def optimizer(self) -> _LRScheduler:
        return self._optimizer

    @property
    def lr_scheduler(self) -> _LRScheduler:
        return self._lr_scheduler

    @staticmethod
    def _get_lr_update_frequency(lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> str:
        if lr_scheduler is not None:
            if lr_scheduler.__class__.__name__ in EPOCH_UPDATE_SCHEDULERS:
                return 'EPOCH_UPDATE'
            else:
                return 'BATCH_UPDATE'
        else:
            return 'NO_UPDATE'

    def update_lr(self, is_batch: bool, metric_value: float = None):
        """Performs learning rate update based on given frequency."""

        # LR_Scheduler should updated after each train batch
        if is_batch and self._lr_update_frequency == 'BATCH_UPDATE':
            self._lr_scheduler.step()

        # LR_Scheduler should updated after each train epoch
        if not is_batch and self._lr_update_frequency == 'EPOCH_UPDATE':
            if self._lr_scheduler.__class__.__name__ in METRIC_BASED_SCHEDULERS:
                self._lr_scheduler.step(metric_value)

            else:
                self._lr_scheduler.step()
        if self._warmup_scheduler is not None:
            self._warmup_scheduler.dampen()

    def get_lr(self) -> float:
        """Returns the learning rate."""
        try:
            lr = self._lr_scheduler.get_last_lr()
            lr = lr[-1]
        except (IndexError, AttributeError):
            lr = self._optimizer.param_groups[0]['lr']
        return lr

    def on_epoch_start(self):
        gc.collect()
        torch.cuda.empty_cache()
        self._evaluator.reset()
        self._network.train()

    def batch_update(self, data_sample: Dict[str, torch.Tensor], loss_kwargs: Dict = None) -> Tuple:
        """
        Performs basic step for training stage.
        Args:
            data_sample: the input data sample, that must contain the input for the network input and the target,
                         it may contain additional staff, such as debug_image and mask
            loss_kwargs: additional keyword arguments to be passed to loss
        Returns:
            loss: the value of loss
            target: the target that is used to compute the loss
            prediction: the prediction of model
        """

        self._optimizer.zero_grad()
        target, prediction = self.get_target_prediction(data_sample)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else dict()
        loss = self._loss(prediction, target, **loss_kwargs)
        if self._regularization is not None:
            loss += self._regularization.forward(self._network)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), 7)
        self._optimizer.step()
        return detach_object((loss, target, prediction))

    def dataloader_loop(self, dataloader: DataLoader, save_indices: Set[int] = None, valid_metric: float = None):
        """Performs loop over dataloader."""
        description = f'    {self._stage_name.upper()[:5]} BATCHES'
        for i, data_sample in enumerate(tqdm(dataloader, desc=description, total=len(dataloader))):
            loss, y_true, y_pred = self.batch_update(data_sample)

            loss = to_device(loss, device='cpu', detach=False)
            self.register_loss(loss)

            self.register_metrics(y_true, y_pred)  # metrics can be calculated on GPU
            (y_true, y_pred) = to_device((y_true, y_pred), device='cpu', detach=False)
            self.register_sample(i, data_sample.get('debug_band'), y_true, y_pred, saving_batches=save_indices)

            self.update_lr(is_batch=True, metric_value=valid_metric)

    def run(self,
            dataloader: DataLoader,
            save_indices: Set[int],
            valid_metric: float = None,
            update_lr: bool = False,
            progress_bar: tqdm = None) -> Dict:
        """Performs Training stage."""
        self.on_epoch_start()

        # updating LR based on previous epoch results
        if update_lr:
            self.update_lr(is_batch=False, metric_value=valid_metric)

        if self._lr_scheduler is not None and progress_bar is not None:
            progress_bar.write(f'       lr={self.get_lr()}')

        if progress_bar is not None:
            progress_bar.write(f' ')

        self.dataloader_loop(dataloader, save_indices=save_indices, valid_metric=valid_metric)
        self._results_keeper['epoch_metrics'] = self._evaluator.compute()
        return self._results_keeper


class SingleBranchValidationEpoch(SingleBranchEpochMixin):
    """Base Validation Epoch."""

    _stage_name = 'validation'

    def __init__(self,
                 network: nn.Module,
                 loss: Loss,
                 evaluator: BaseEvaluator,
                 device: str = 'cpu',
                 verbose: bool = True):
        """
        Args:
            network: the PyTorch model
            loss: the Loss object
            evaluator: the evaluator object
            device: cpu or cuda
            verbose: if True will show intermediate results
        """

        self._network = network
        self._loss = loss
        self._evaluator = evaluator
        self._verbose = verbose
        self._device = device

        self._results_keeper = self.get_results_keeper()
        self._to_device()

    def on_epoch_start(self):
        gc.collect()
        torch.cuda.empty_cache()
        self._evaluator.reset()
        self._network.eval()

    def batch_update(self, data_sample: Dict[str, torch.Tensor], loss_kwargs: Dict = None):
        """
        Performs basic step for validation stage.
        Args:
            data_sample: the input data sample, that must contain the input for the network input and the target,
                        it may contain additional staff, such as debug_image and mask
            loss_kwargs: additional keyword arguments to be passed to loss
        Returns:
            loss: the value of loss
            target: the target that is used to compute the loss
            prediction: the prediction of model
       """
        with torch.no_grad():
            target, prediction = self.get_target_prediction(data_sample)
            loss_kwargs = loss_kwargs if loss_kwargs is not None else dict()
            loss = self._loss(prediction, target, **loss_kwargs)
        return loss, target, prediction

    def dataloader_loop(self, dataloader: DataLoader, save_indices: Set[int] = None):
        """Performs loop over dataloader."""
        description = f'    {self._stage_name.upper()[:5]} BATCHES'
        for i, data_sample in enumerate(tqdm(dataloader, desc=description, total=len(dataloader))):
            loss, y_true, y_pred = self.batch_update(data_sample)

            loss = to_device(loss, device='cpu', detach=False)
            self.register_loss(loss)

            self.register_metrics(y_true, y_pred)  # metrics can be calculated on GPU
            (y_true, y_pred) = to_device((y_true, y_pred), device='cpu', detach=False)
            self.register_sample(i, data_sample.get('debug_band'), y_true, y_pred, saving_batches=save_indices)

    def run(self,
            dataloader: DataLoader,
            save_indices: Set[int],
            progress_bar: tqdm = None) -> Dict:
        """Performs Validation run."""
        self.on_epoch_start()

        if progress_bar is not None:
            progress_bar.write(f' ')

        self.dataloader_loop(dataloader, save_indices=save_indices)
        self._results_keeper['epoch_metrics'] = self._evaluator.compute()
        return self._results_keeper
