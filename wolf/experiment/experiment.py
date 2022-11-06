import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import BaseExperimentConfig
from .experiment_logging import BaseExperimentLogger
from .parser import BaseExperimentConfigParser
from ..epoch import SingleBranchTrainEpoch, SingleBranchValidationEpoch
from ..logger import logger

__all__ = ['BaseExperiment', 'run_experiment']


class BaseExperiment:
    """Defines the Base Class for running different experiments."""

    DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

    GPU_TEMPERATURE_THRESHOLD = 80

    CONFIG_KLASS = BaseExperimentConfig
    CONFIG_PARSER_KLASS = BaseExperimentConfigParser

    def __init__(self, config: str or Dict):
        """
        Args:
            config: the path to JSON file with configs or the dict of configs
        """
        self.config = self.CONFIG_KLASS(config)
        self.config_parser = self.CONFIG_PARSER_KLASS(self.config)

        self.logger = self.get_logger_obj()
        self.logger.register_run_config(self.config)

        self._train_batches, self._valid_batches = None, None

        # buffer for selection metric
        self._selection_metric = self._get_selection_metric()
        self._early_stopping = self.config_parser.get_early_stopping()
        self._early_stopping.adjust_direction(self._selection_metric['direction'])

    def get_logger_obj(self) -> BaseExperimentLogger:
        """Initializes BaseExperimentLogger instance with selecting unique folder to store data."""

        exp_id = f"{datetime.now().strftime(self.DATE_FORMAT)}-{self.config.experiment.name}"
        try:
            saving_dir = Path(self.config.experiment.kwargs.save_path).joinpath(exp_id)
            saving_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            logger.error(f"Make sure the provided experiment_id={exp_id} is unique.")
            logger.error(e, stack_info=True)
            raise e
        else:
            logger.info(f'Experiment results will be saved in "{saving_dir}" directory.')

        n = self.config_parser.get_n_debug_samples()
        logger.info(f'N={n} images from each batch will be saved.')

        logger_klass = self.config_parser.get_logger_class()
        return logger_klass(saving_dir, n, clearml_config=self.config.experiment.kwargs.get('clearml'))

    def _get_selection_metric(self) -> Dict:
        """Returns the name of the selection metric."""
        selection_config = self.config.experiment.kwargs.get("selection_metric", {})
        metric_name = selection_config.get("name", self.config.loss.name)  # default loss
        direction = selection_config.get("direction", "min")  # default minimize

        metric_factor = 1 if direction == "min" else -1
        metric_value = np.inf * metric_factor

        logger.info(f"Training will {'minimize' if direction == 'min' else 'maximize'} {metric_name} metric on "
                    f"validation data, starting value='{metric_value}', factor={metric_factor}.")
        return {
            'name': metric_name,
            'history': [{'epoch': -1, 'value': metric_value}],
            'factor': metric_factor,
            'direction': direction
        }

    def generate_inference_config(self) -> Dict:
        """This method aims to generate inference config from given running config.
         This method should be implemented in subclasses."""
        raise NotImplementedError

    def _register_inference_config(self):
        """Registers an inference config."""
        try:
            self.logger.register_inference_config(self.generate_inference_config())
        except NotImplementedError:
            logger.info(f'Inference config is not generated, please implement `generate_inference_config` method.')

    def run(self):
        """Runs the experiment."""
        try:
            logger.info(f"Starting experiment: Loading Model, Data, Loss and Metrics.")
            train_epoch, valid_epoch = self._get_epoch_objects()
            train_dataset, valid_dataset, train_loader, valid_loader = self.config_parser.get_data_loaders()
            self._train_batches, self._valid_batches = len(train_loader), len(valid_loader)
            train_saving_batches, valid_saving_batches = self._get_saving_batches(train_loader, valid_loader)
            self.logger.register_dataset(train_dataset, mode='train')
            self.logger.register_dataset(valid_dataset, mode='valid')
            logger.info(f'Number of train: images = {len(train_dataset)}, batches = {self._train_batches}')
            logger.info(f'Number of valid: images = {len(valid_dataset)}, batches = {self._valid_batches}')
            logger.info(
                f"Number of model parameters = "
                f"{sum(p.flatten().shape[0] for p in train_epoch.network.parameters() if p.requires_grad) :,}"
            )
            logger.info(f"Model, Data, Loss and Metrics are loaded successfully. Starting training.")
        except Exception as e:
            logger.error(f"Training is interrupted due to error: {e}.", stack_info=True, exc_info=True)
            self.logger.clean_up(stopped_by_user=False)
        else:

            progress_bar = self._get_progress_bar()
            for epoch in progress_bar:
                self.check_gpu_temperature(progress_bar=progress_bar)
                try:
                    progress_bar.write('')
                    progress_bar.write(f'   Running epoch: {epoch}')

                    verbose = epoch % self.config.experiment.kwargs.verbose_epochs == 0

                    if self._train_batches:
                        train_epoch.run(
                            train_loader, train_saving_batches if verbose else None,
                            valid_metric=self._selection_metric['history'][-1]['value'],
                            update_lr=epoch != 1,
                            progress_bar=progress_bar
                        )

                    if self._valid_batches:
                        valid_epoch.run(valid_loader, valid_saving_batches if verbose else None)

                    self._register_epoch_results(epoch, train_epoch, valid_epoch, progress_bar)

                    if self._early_stopping.early_stop:
                        break

                    progress_bar.write('')

                except KeyboardInterrupt:
                    logger.info("Training is interrupted by user.")
                    self.logger.clean_up(stopped_by_user=True)
                    break

                except Exception as e:
                    logger.error(f"Training is interrupted due to error: {e}.", stack_info=True, exc_info=True)
                    self.logger.clean_up(stopped_by_user=False)
                    break

            self.logger.finalize()
            logger.info(f'Training results are saved in "{self.logger.saving_dir}" .')

    def _get_epoch_objects(self) -> Tuple[SingleBranchTrainEpoch, SingleBranchValidationEpoch]:
        """Initializes all necessary objects and returns Train and Validation Epoch instances."""
        logger.info(f"Starting experiment: Loading Model, Data, Loss and Metrics.")
        network = self.config_parser.get_model()
        optimizer = self.config_parser.get_optimizer(network)
        lr_scheduler = self.config_parser.get_lr_scheduler(optimizer)
        warmup_scheduler = self.config_parser.get_warmup_scheduler(optimizer)
        loss = self.config_parser.get_loss()
        regularization = self.config_parser.get_regularizer()
        device = self.config_parser.get_device()

        train_evaluator = self.config_parser.get_evaluator(device=device)
        valid_evaluator = self.config_parser.get_evaluator(device=device)
        train_epoch_cls, valid_epoch_cls = self.config_parser.get_epochs_classes()

        train_epoch = train_epoch_cls(
            network, loss, train_evaluator, optimizer, lr_scheduler,
            warmup_scheduler=warmup_scheduler, regularization=regularization, device=device, verbose=True
        )
        valid_epoch = valid_epoch_cls(network, loss, valid_evaluator, device=device, verbose=True)
        return train_epoch, valid_epoch

    def _get_progress_bar(self) -> tqdm:
        """Returns tqdm progress bar."""
        return tqdm(
            range(1, self.config.experiment.kwargs.epochs + 1),
            total=self.config.experiment.kwargs.epochs, desc='Epochs', position=0, leave=True
        )

    def _get_saving_batches(self, train_loader: DataLoader, valid_loader: DataLoader) -> Tuple[set, set]:
        """Returns random indexes for saving samples from training and validation datasets."""
        # saving self.N_SAMPLES_TO_SAVE samples during each successful epoch
        n = self.config_parser.get_n_debug_samples()
        if n == 0:
            return set(), set()
        else:
            size = max(1, n // self.config.experiment.kwargs.batch_size)
            logger.info(f'N={size} batches are selected for saving sample images.')
            train_saving_batches = set(np.random.randint(0, len(train_loader), size=size))
            valid_saving_batches = set(np.random.randint(0, len(valid_loader), size=size))
            return train_saving_batches, valid_saving_batches

    def _register_epoch_results(self,
                                epoch: int,
                                train_epoch: SingleBranchTrainEpoch,
                                valid_epoch: SingleBranchValidationEpoch,
                                progress_bar: tqdm):
        """Registers epoch results using ExperimentResult object."""
        if self._train_batches:
            # train step
            train_loss_metrics = train_epoch.get_epoch_metrics()
            lr = train_epoch.get_lr()
            self.logger.register_epoch_results(epoch, 'train', train_loss_metrics, lr=lr, progress_bar=progress_bar)

            if not self._valid_batches:
                # registering network after each batch if the validation is not given
                self.logger.register_model(epoch, train_epoch.network, train_epoch.optimizer, train_epoch.loss)

        if self._valid_batches:
            # validation step
            valid_loss_metrics = valid_epoch.get_epoch_metrics()
            self.logger.register_epoch_results(epoch, 'valid', valid_loss_metrics, progress_bar=progress_bar)

            name, factor = self._selection_metric['name'], self._selection_metric['factor']
            new_value, current_value = valid_loss_metrics[name], self._selection_metric['history'][-1]['value']
            if new_value * factor < current_value * factor:
                progress_bar.write('')
                progress_bar.write(f'      Saving model: {name}: "{current_value:.4}" -> "{new_value:.4}"')
                self._selection_metric['history'].append({'epoch': epoch, 'value': new_value})
                self.logger.register_selection_metric(self._selection_metric)
                self.logger.register_model(epoch, train_epoch.network, train_epoch.optimizer, train_epoch.loss)
                self.logger.register_epoch_debug_samples(epoch, 'train', train_epoch.results_keeper['images'])
                self.logger.register_epoch_debug_samples(epoch, 'valid', valid_epoch.results_keeper['images'])
                # self.logger.register_model_params_histograms(epoch, train_epoch.network)
            # updating early stopping with selection metric new value
            self._early_stopping(new_value, progress_bar=progress_bar)

        # registering inference config, after the first epoch
        if epoch == 1:
            self._register_inference_config()

    def check_gpu_temperature(self, progress_bar: tqdm = None):
        """If GPU temp is more than threshold, sleeps 60sec"""
        if self.config.experiment.kwargs.get('check_gpu_temperature', False):
            if "cuda" in self.config.experiment.kwargs['device'] and torch.cuda.is_available():
                from nvgpu.list_gpus import device_statuses

                while True:
                    gpu_max_temp = max(x['temperature'] for x in device_statuses())
                    if gpu_max_temp >= self.GPU_TEMPERATURE_THRESHOLD:
                        if progress_bar is not None:
                            progress_bar.write(f'          ---sleeping 1 min to enable GPU cool down--- ')
                        time.sleep(60 * 1)  # sleeping 1 minutes
                    else:
                        break


def run_experiment(config_path: str):
    """Runs experiment with given config."""
    experiment_obj = BaseExperiment(config_path)
    experiment_obj.run()
