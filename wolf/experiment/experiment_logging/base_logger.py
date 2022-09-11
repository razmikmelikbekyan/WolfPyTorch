import json
import logging
import shutil
import signal
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from multiprocessing import Process, Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from clearml import Task, Logger, OutputModel
from clearml.storage.helper import StorageHelper
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from ..config import BaseExperimentConfig
from ...dataset import WolfDataset
from ...logger import logger
from ...losses import Loss
from ...plotting_service.classification import plot_confusion_matrix

__all__ = ['BaseExperimentLogger']

logging.getLogger("clearml.storage").setLevel("WARNING")


class ConfigEncoder(json.JSONEncoder):
    """Special class for encoding Config object."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _mgr_init():
    """Special initializer for SyncManager to ignore signals."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)


class BaseExperimentLogger:
    """Base Class for storing and saving experiments results."""

    RUN_CONFIG_FILENAME = 'run_config.JSON'
    INFERENCE_CONFIG_FILENAME = 'inference_config.JSON'
    RESULTS_FILENAME = 'results.JSON'
    SELECTION_METRIC_FILENAME = 'selection_metric.JSON'
    MODEL_FILENAME = 'model.pt'
    MODEL_WEIGHTS_FILENAME = 'model_weights.pt'
    IMAGES_FOLDER = 'images'
    IMAGES_FILENAME = 'images.csv'
    DATASET_FILENAME = 'dataset.pkl'

    # TENSORBOARD_FOLDER = 'tensorboard'

    @property
    @abstractmethod
    def FIG_SIZE(self) -> Tuple[int, int]:
        """Defines the figure size for saving debug images."""

    def __init__(self, saving_dir: str or Path, n_debug_samples: int, clearml_config: Optional[Dict] = None):
        """
        Args:
            saving_dir: the directory to store all results
            n_debug_samples: the number of debug samples to save for each epoch
            clearml_config: the config for clearml
        """

        self.saving_dir = Path(saving_dir)
        self.n_debug_samples = n_debug_samples
        self._cleaned_up = False

        self.saving_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.saving_dir.joinpath(self.IMAGES_FOLDER)
        self.images_dir.mkdir(exist_ok=True)

        self._metrics_keeper = defaultdict(list)

        # multiprocessing staff
        manager = SyncManager()
        manager.start(_mgr_init)  # fire up the child manager process
        self._debug_samples_mp_manager = manager.list()
        self._debug_samples_processes = []

        self.clearml_task: Task = None
        self.clearml_logger: Logger = None
        self.clearml_model: OutputModel = None
        if clearml_config:
            clearml_config['auto_connect_frameworks'] = False
            Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
            self.clearml_task = Task.init(task_name=self.saving_dir.name, **clearml_config)
            self.clearml_logger = self.clearml_task.get_logger()
            self.clearml_model = OutputModel(self.clearml_task)
            destination = self.clearml_task.get_output_destination()
            self.clearml_logger.set_default_upload_destination(f'{destination}')

    @property
    def folder(self) -> Path:
        """Returns the folder where the results are stored."""
        return self.saving_dir

    def clean_up(self):
        """Removes logged data due to errors."""
        if not self.results_path.exists():
            shutil.rmtree(self.saving_dir)
            self._cleaned_up = True
            if self.clearml_task is not None:
                self.clearml_task.delete(raise_on_error=True)
        else:
            logger.warning(f"Failed to remove experiment folder, since it contains results data.")

    @property
    def cleaned_up(self) -> bool:
        """If clean_up is called returns True"""
        return self._cleaned_up

    @property
    def run_config_path(self) -> Path:
        """Returns the run configfile path."""
        return self.saving_dir.joinpath(self.RUN_CONFIG_FILENAME)

    @property
    def inference_config_path(self) -> Path:
        """Returns the inference configfile path."""
        return self.saving_dir.joinpath(self.INFERENCE_CONFIG_FILENAME)

    @property
    def results_path(self) -> Path:
        """Returns the JSON file path, where the metrics will be stored."""
        return self.saving_dir.joinpath(self.RESULTS_FILENAME)

    @property
    def selection_metric_path(self) -> Path:
        """Returns the JSON file path, where the selection metric values will be stored."""
        return self.saving_dir.joinpath(self.SELECTION_METRIC_FILENAME)

    @property
    def model_path(self) -> Path:
        """Returns the file path, where the model will be stored."""
        return self.saving_dir.joinpath(self.MODEL_FILENAME)

    @property
    def model_weights_path(self) -> Path:
        """Returns the file path, where the model will be stored."""
        return self.saving_dir.joinpath(self.MODEL_WEIGHTS_FILENAME)

    @property
    def train_dataset_path(self) -> Path:
        """Returns the pickle file path, where the dataset will be saved."""
        return self.saving_dir.joinpath(f'train_{self.DATASET_FILENAME}')

    @property
    def valid_dataset_path(self) -> Path:
        """Returns the pickle file path, where the dataset will be saved."""
        return self.saving_dir.joinpath(f'valid_{self.DATASET_FILENAME}')

    def _save_debug_samples_data(self):
        """After each epoch saves saved images info."""
        df = pd.DataFrame(list(self._debug_samples_mp_manager))
        if df.empty:
            logger.debug(f"No information about saved images")
        else:
            if self.saving_dir.exists():
                df = df.sort_values(by=['epoch', 'mode', 'batch', 'image_index'])
                saving_path = self.saving_dir.joinpath(self.IMAGES_FILENAME)
                df.to_csv(saving_path)
                logger.debug(f"Information about saved images is saved in {saving_path} file.")
                if self.clearml_logger is not None:
                    self.clearml_logger.report_table("DebugSamples", "all", csv=saving_path.as_posix())

    def finalize(self):
        """Waits until all processes are finished."""
        for p in self._debug_samples_processes:
            p.join()

        self._save_debug_samples_data()

        if self.clearml_model is not None:
            try:
                self.register_best_epoch_results()

                self.clearml_model.update_weights(
                    weights_filename=self.model_weights_path.as_posix(),
                    auto_delete_file=False
                )
                # waiting to finish all threads
                pool = StorageHelper._upload_pool
                if pool:
                    pool.close()
                    pool.join()
            except Exception as e:
                logger.warning(f"Failed to log final results into clearml, due to error: {e}")

            try:
                self.clearml_task.close()
            except Exception as e:
                logger.warning(f"Failed to close clearml task due to error: {e}.", exc_info=True)
            else:
                logger.info(f"clearml task is closed.")

    def register_run_config(self, config: BaseExperimentConfig):
        """Registers the experiment config."""
        with open(self.run_config_path, 'w') as f:
            json.dump(config.all_configs, f, cls=ConfigEncoder)

        if self.clearml_task is not None:
            self.clearml_task.connect_configuration(name='run_config', configuration=config.all_configs)
            self.clearml_model.update_design(config_dict=config.model)
            hparams = config.get_hyper_params_to_log()
            self.clearml_task.set_parameters_as_dict(hparams)

            epochs = hparams['early_stopping_patience']
            self.clearml_logger.set_default_debug_sample_history(epochs)

        print()
        for k, v in config.all_configs.items():
            logger.info(f'{k} config: {v}')
            print()

        logger.debug(f"Experiment run config is saved in {self.run_config_path} file.")

    def register_inference_config(self, config: Dict):
        """Registers the inference config."""
        if not self._cleaned_up:
            with open(self.inference_config_path, 'w') as f:
                json.dump(config, f, cls=ConfigEncoder)

            if self.clearml_task is not None:
                self.clearml_task.upload_artifact(name='inference_config', artifact_object=config)

            logger.debug(f"Experiment inference config is saved in {self.inference_config_path} file.")

    @staticmethod
    def _format_results(results: Dict[str, float or int]) -> str:
        return ' | '.join(
            [
                f'{k} - {v:.4}' for k, v in results.items()
                if isinstance(v, (float, int, np.floating, np.integer)) and not np.isnan(v)
            ]
        )

    def _clearml_log_metrics(self, metrics: Dict, epoch_mode: str, epoch: int):
        """Uses clearml to log metrics."""
        if self.clearml_logger is not None:
            for k, v in metrics.items():
                if isinstance(v, (float, int, np.floating, np.integer)):
                    self.clearml_logger.report_scalar(k, epoch_mode, v, epoch)
                elif isinstance(v, (list, np.ndarray)) and k in ('confusion_matrix', 'class_names'):
                    pass
                else:
                    raise TypeError(f"Metrics should be scalar or matrix, got {type(v)}")

            cm = metrics.get('confusion_matrix')
            class_names = metrics.get('class_names')
            if cm is not None:
                fig = plot_confusion_matrix(np.array(cm), class_names, annotate_samples=False, return_figure=True)
                self.clearml_logger.report_matplotlib_figure(
                    title=f'{epoch_mode}/CFM',
                    series=f'epoch={epoch}',
                    figure=fig,
                    iteration=epoch,
                    report_image=False,
                    report_interactive=True,
                )
                plt.close(fig=fig)

    def register_epoch_results(self,
                               epoch: int,
                               epoch_mode: str,
                               epoch_loss_metrics: Dict,
                               lr: float = None,
                               progress_bar: tqdm = None):
        """Registers the experiment single epoch results."""
        if progress_bar is not None:
            progress_bar.write("")
            progress_bar.write(f'      Epoch {epoch_mode.upper()} Results: {self._format_results(epoch_loss_metrics)}')

        serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in epoch_loss_metrics.items()}
        if lr:
            serializable['lr'] = lr

        self._clearml_log_metrics(serializable, epoch_mode, epoch)

        serializable['epoch'] = epoch
        self._metrics_keeper[epoch_mode].append(serializable)

        with open(self.results_path, 'w') as f:
            json.dump(dict(self._metrics_keeper), f, cls=NumpyEncoder)
        logger.debug(f"Experiment results are saved in {self.results_path} file.")

    def register_selection_metric(self, selection_metric: Dict):
        """Registers the selection_metric data."""
        with open(self.selection_metric_path, 'w') as f:
            json.dump(selection_metric, f, cls=NumpyEncoder)
        logger.debug(f"Experiment selection metric data is saved in {self.selection_metric_path} file.")

    def register_best_epoch_results(self):
        """Registers the best epoch results."""
        with open(self.selection_metric_path) as f:
            selection_metric = json.load(f)

        selection_metric = pd.DataFrame(selection_metric['history'])
        best_epoch = int(selection_metric['epoch'].max())

        logger.info(f"The best_epoch={best_epoch}. Logging its values.")
        self.clearml_logger.report_text(f'BestEpoch={best_epoch}', print_console=False)
        for mode, mode_metrics in self._metrics_keeper.items():
            [best_epoch_metrics] = [item for item in mode_metrics if item['epoch'] == best_epoch]
            best_epoch_metrics = deepcopy(best_epoch_metrics)
            best_epoch_metrics.pop('epoch')
            serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_epoch_metrics.items()}
            serializable.pop('lr', None)
            self._clearml_log_metrics(serializable, f'best_{mode}', best_epoch)
        logger.info("Best Epoch results are logged.")

    def register_model(self, epoch: int, network: nn.Module, optimizer: Optimizer, loss: Loss):
        """Registers the model."""
        try:
            state_dict = network.module.state_dict()
        except AttributeError:
            state_dict = network.state_dict()

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            },
            self.model_path,
        )
        torch.save(state_dict, self.model_weights_path)
        logger.debug(f"Training checkpoint is saved in {self.model_path} file.")
        logger.debug(f"Model weights are saved in {self.model_weights_path} file.")

    def register_dataset(self, dataset: WolfDataset, mode: str):
        """Registers dataset DataFrame."""
        if mode == 'train':
            saving_path = str(self.train_dataset_path)
        elif mode == 'valid':
            saving_path = str(self.valid_dataset_path)
        else:
            raise ValueError(f"mode must be train or valid, got: '{mode}'")

        dataset.df.to_pickle(saving_path)
        logger.debug(f"{mode.upper()} dataset pickle is saved in {saving_path} file.")

        if self.clearml_task is not None:
            self.clearml_task.upload_artifact(f"{mode}_dataset", saving_path)

    def register_epoch_debug_samples(self, epoch: int, epoch_mode: str, epoch_images: Dict):
        """Registers the experiment single epoch results."""
        if self.n_debug_samples <= 0:
            return

        for batch, batch_images in epoch_images.items():
            if 'debug_image' not in batch_images or batch_images['debug_image'] is None:
                continue
            else:
                p = Process(
                    target=self.save_epoch_images,
                    kwargs=dict(
                        images_dir=self.images_dir,
                        epoch_mode=epoch_mode,
                        epoch_number=epoch,
                        batch_number=batch,
                        n_images_to_save=self.n_debug_samples,
                        info=dict(mode=epoch_mode, epoch=epoch, batch=batch),
                        mp_manager=self._debug_samples_mp_manager,
                        clearml_logger=self.clearml_logger,
                        **batch_images
                    )
                )
                p.start()
                self._debug_samples_processes.append(p)

    @classmethod
    def save_epoch_images(cls,
                          input_image: np.ndarray,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          info: Dict,
                          mp_manager: Manager,
                          mode: str = None,
                          epoch_number: int = None,
                          batch_number: int = None,
                          images_dir: str = None,
                          clearml_logger: Optional[Logger] = None,
                          n_images_to_save: int = 10):
        """Please implement this method in subclasses, depending on the task."""
        raise NotImplementedError

    @staticmethod
    def save_image(image: np.ndarray, saving_path: str or Path, as_uint8: bool = True) -> None:
        """Saves single debug image."""
        if as_uint8:
            cv2.imwrite(str(saving_path), (image * 255).astype(np.uint8))
        else:
            cv2.imwrite(str(saving_path), image)
