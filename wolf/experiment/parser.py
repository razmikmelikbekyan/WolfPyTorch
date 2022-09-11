from typing import Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, RandomSampler

from .config import BaseExperimentConfig
from .early_stopping import EarlyStopping
from .experiment_logging import BaseExperimentLogger, get_experiment_logger
from ..dataset import WolfDataset
from ..epoch import SingleBranchTrainEpoch, SingleBranchValidationEpoch, get_epochs
from ..losses import get_loss, Loss
from ..metrics import BaseEvaluator, get_evaluator
from ..models import get_model
from ..models.helpers.weights.device import model_to_device
from ..optimization import get_optimizer, get_lr_scheduler, get_warmup_scheduler
from ..regularization import get_regularizer, WeightsRegularization


class BaseExperimentConfigParser:
    """Base Class for parsing given BaseExperimentConfig.
    Up to your needs you can inherit this class and create your own parsers.
    """
    DEFAULT_N_DEBUG_SAMPLES = 20  # the number of debug samples to save during each epoch

    def __init__(self, config_obj: BaseExperimentConfig):
        if not isinstance(config_obj, BaseExperimentConfig):
            raise ValueError(f"Given config object must be an instance of {BaseExperimentConfig}, "
                             f"got: {type(config_obj)}.")

        self.config = config_obj

    def get_device(self):
        """Returns the device to be used further."""
        device: str = self.config.experiment.kwargs.get('device', 'cpu')

        if device == 'cpu':
            return 'cpu'
        elif device.startswith('cuda'):
            cuda_is_available = torch.cuda.is_available()
            if not cuda_is_available:
                return 'cpu'
            else:
                available_devices = torch.cuda.device_count()
                device_ids = self.config.experiment.kwargs.get('device_ids', None)
                if device_ids is not None:
                    device_ids = sorted(device_ids)[:available_devices]
                else:
                    device_ids = list(range(available_devices))
                return f'cuda:{device_ids[0]}'
        else:
            raise ValueError(f"Device must be 'cpu' or 'cuda', got: {device}")

    def get_n_debug_samples(self) -> int:
        """Returns the number of debug samples to save during each epoch"""
        n_debug_samples = self.config.experiment.kwargs.get('n_debug_samples', self.DEFAULT_N_DEBUG_SAMPLES)
        if n_debug_samples > self.config.experiment.kwargs.batch_size:
            n_debug_samples = self.config.experiment.kwargs.batch_size
        return n_debug_samples

    def get_datasets(self) -> Tuple[WolfDataset, WolfDataset]:
        """Returns the train and the valid dataset objects."""
        return WolfDataset(**self.config.train_data.kwargs), WolfDataset(**self.config.valid_data.kwargs),

    def get_data_loaders(self) -> Tuple[WolfDataset, WolfDataset, DataLoader, DataLoader]:
        """Returns the train and valid data loaders of experiment."""
        train_dataset, valid_dataset = self.get_datasets()

        train_steps_per_epoch = self.config.experiment.kwargs.get('train_steps_per_epoch', None)
        if train_steps_per_epoch:
            train_sampler = RandomSampler(
                train_dataset,
                replacement=True,
                num_samples=train_steps_per_epoch * self.config.experiment.kwargs.batch_size
            )
        else:
            train_sampler = None

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.experiment.kwargs.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.config.experiment.kwargs.num_workers,
            pin_memory=False,
            sampler=train_sampler
        )

        valid_steps_per_epoch = self.config.experiment.kwargs.get('valid_steps_per_epoch', None)
        if valid_steps_per_epoch:
            valid_sampler = RandomSampler(
                valid_dataset,
                replacement=True,
                num_samples=valid_steps_per_epoch * self.config.experiment.kwargs.batch_size
            )
        else:
            valid_sampler = None

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config.experiment.kwargs.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.experiment.kwargs.num_workers,
            pin_memory=False,
            sampler=valid_sampler
        )

        return train_dataset, valid_dataset, train_loader, valid_loader

    def get_model(self) -> nn.Module:
        """Returns the experiment network."""
        model = get_model(high_level_architecture=self.config.experiment.kwargs.high_level_architecture,
                          task=self.config.experiment.kwargs.task,
                          model_name=self.config.model.name,
                          **self.config.model.kwargs)
        device_ids = self.config.experiment.kwargs.get('device_ids', None)
        return model_to_device(model, device='cuda', multi_gpu=True, device_ids=device_ids)

    def get_loss(self) -> Loss:
        """Returns the loss object based on config."""
        return get_loss(high_level_architecture=self.config.experiment.kwargs.high_level_architecture,
                        task=self.config.experiment.kwargs.task,
                        loss_name=self.config.loss.name,
                        **self.config.loss.kwargs)

    def get_regularizer(self) -> WeightsRegularization:
        """Returns the RegularizationLoss object based on config."""
        try:
            return get_regularizer(self.config.experiment.kwargs.regularization.name,
                                   **self.config.experiment.kwargs.regularization.kwargs)
        except AttributeError:
            return None

    def get_evaluator(self, device: str or torch.device = 'cpu') -> BaseEvaluator:
        """Returns the list of metric objects based on config."""
        return get_evaluator(high_level_architecture=self.config.experiment.kwargs.high_level_architecture,
                             task=self.config.experiment.kwargs.task,
                             **self.config.evaluator.kwargs,
                             device=device
                             )

    def get_optimizer(self, network: nn.Module) -> Optimizer:
        """Returns the optimizer object."""
        return get_optimizer(self.config.optimizer.name, network, **self.config.optimizer.kwargs)

    def get_lr_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        """Returns the Learning rate scheduler object."""
        lr_scheduler_config = self.config.optimizer.get('lr_scheduler')
        if not lr_scheduler_config:
            return
        else:
            return get_lr_scheduler(lr_scheduler_config.name, optimizer, **lr_scheduler_config.kwargs)

    def get_warmup_scheduler(self, optimizer: Optimizer):
        """Returns the warmup scheduler object."""
        warmup_scheduler = self.config.optimizer.get('warmup_scheduler')
        if not warmup_scheduler:
            return
        else:
            return get_warmup_scheduler(warmup_scheduler.name, optimizer, **warmup_scheduler.kwargs)

    def get_early_stopping(self) -> EarlyStopping:
        """Returns the EarlyStopping object."""
        config = self.config.experiment.kwargs.get('early_stopping', None)
        config = config if config is not None else {'patience': np.inf}
        return EarlyStopping(**config)

    def get_epochs_classes(self) -> Tuple[Type[SingleBranchTrainEpoch], Type[SingleBranchValidationEpoch]]:
        """Returns the train and validation epoch classes."""
        return get_epochs(high_level_architecture=self.config.experiment.kwargs.high_level_architecture,
                          task=self.config.experiment.kwargs.task)

    def get_logger_class(self) -> Type[BaseExperimentLogger]:
        """Returns the experiment logger class."""
        return get_experiment_logger(self.config.experiment.kwargs.task)
