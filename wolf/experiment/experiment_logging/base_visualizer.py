import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from termcolor import colored

from .base_logger import BaseExperimentLogger
from ..config import BaseExperimentConfig
from ...logger import logger
from ...plotting_service.image import plot_rgb_image

__all__ = [
    'BaseExperimentsResultVisualizer'
]


class BaseExperimentsResultVisualizer:
    """Base class for view ExperimentResults."""

    _figsize: Tuple[int, int]

    def __init__(self, result_dir: str or Path):
        """

        Args:
            result_dir: the directory where the experiment results are stored.
        """
        self._results_dir = Path(result_dir) if isinstance(result_dir, str) else result_dir
        self._images_dir = self._results_dir.joinpath(BaseExperimentLogger.IMAGES_FOLDER)
        self._images_csv_path = self._results_dir.joinpath(BaseExperimentLogger.IMAGES_FILENAME)
        self._run_config_path = self._results_dir.joinpath(BaseExperimentLogger.RUN_CONFIG_FILENAME)
        self._results_json_path = self._results_dir.joinpath(BaseExperimentLogger.RESULTS_FILENAME)
        self._selection_metric_json_path = self._results_dir.joinpath(BaseExperimentLogger.SELECTION_METRIC_FILENAME)
        self._model_path = self._results_dir.joinpath(BaseExperimentLogger.MODEL_FILENAME)
        self._train_dataset_path = self._results_dir.joinpath(f'train_{BaseExperimentLogger.DATASET_FILENAME}')
        self._valid_dataset_path = self._results_dir.joinpath(f'valid_{BaseExperimentLogger.DATASET_FILENAME}')

        self._run_config = BaseExperimentConfig(str(self._run_config_path))
        self._task = self._run_config.experiment.kwargs.task

    @staticmethod
    def get_metrics(json_path: str or Path, mode: str) -> pd.DataFrame:
        """Returns the saved train or validation metrics."""
        with open(json_path) as f:
            results = json.load(f)
        return pd.DataFrame.from_dict(results[mode])

    @staticmethod
    def get_selection_metric(json_path: str or Path) -> pd.DataFrame:
        """Returns the saved train and validation metrics."""
        with open(json_path) as f:
            selection_metric = json.load(f)
        return pd.DataFrame(selection_metric['history']).rename(columns={'value': selection_metric['name']})

    def _get_scalar_metrics(self) -> List[str]:
        """Returns the list of scalar metrics."""
        return self.train_metrics.select_dtypes(include=['number']).columns.tolist()

    @property
    def train_metrics(self) -> pd.DataFrame:
        """Returns the dataframe with training results from each epoch."""
        return self.get_metrics(self._results_json_path, 'train')

    @property
    def valid_metrics(self) -> pd.DataFrame:
        """Returns the dataframe with training results from each epoch."""
        return self.get_metrics(self._results_json_path, 'valid')

    @property
    def selection_metric(self) -> pd.DataFrame:
        """Returns the selection metric dataframe."""
        return self.get_selection_metric(self._selection_metric_json_path)

    @property
    def images_df(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self._images_csv_path, index_col=0)
        except FileNotFoundError:
            logger.warning("Image data is not yet finished saving, trying to recover from saved images.")
            try:
                images = list(self._images_dir.iterdir())
                images = self._image_paths_to_dataframe(images)
            except Exception as e:
                logger.info(f"Failed to recover image data from saved images, due to error: {e}")
                logger.warning("Image data is not yet finished saving, please try after training is finished.")
            else:
                return images

    @staticmethod
    def _image_paths_to_dataframe(image_paths: List[Path], real_paths: List[str] = None) -> pd.DataFrame:
        """Converts image paths to dataframe."""
        images = [(str(x), *x.stem.split('_')) for x in image_paths]
        images = [
            {'image_path': x[0], 'mode': x[1], 'epoch': int(x[2]), 'batch': int(x[3]), 'image_index': int(x[4])}
            for x in images
        ]
        images = pd.DataFrame(images)
        if real_paths:
            images['image_path'] = real_paths
        images = images.sort_values(by=['epoch', 'batch', 'image_index'])
        return images

    @property
    def available_epochs_images(self) -> List[int]:
        """Returns the list of epochs for which debug images have been saved."""
        df = self.images_df
        if df is not None:
            return sorted(set(df['epoch']))

    @property
    def run_config(self) -> BaseExperimentConfig:
        """Returns the ConfigObject used for running experiment."""
        return self._run_config

    @property
    def train_dataset(self) -> pd.DataFrame:
        """Returns the pandas DataFrame with the train images used for running experiment."""
        return pd.read_pickle(self._train_dataset_path).copy()

    @property
    def valid_dataset(self) -> pd.DataFrame:
        """Returns the pandas DataFrame with the valid images used for running experiment."""
        return pd.read_pickle(self._valid_dataset_path).copy()

    @staticmethod
    def plot_scalar(train_values: pd.Series, valid_values: pd.Series, best_epoch: int = None):
        """Plots scalar metric or loss values over epochs."""
        plt.figure(figsize=(10, 8))
        plt.title(train_values.name, size=14, weight='bold')
        sns.lineplot(x=range(1, len(train_values) + 1), y=train_values.values, label='train', color='orange')
        sns.lineplot(x=range(1, len(valid_values) + 1), y=valid_values.values, label='valid', color='darkgreen')
        if best_epoch is not None:
            plt.axvline(x=best_epoch, color='red', label='best_epoch', linestyle='--', linewidth=1.)

        plt.xlabel('epochs', size=12)
        plt.ylabel('metric value', size=12)
        plt.legend(loc='upper left')
        plt.tight_layout()
        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.show()

    def plot_all_scalar_metrics(self, metrics: List[str] = None):
        """Plots all scalar metrics over epochs."""

        sns.set(style="ticks", context='talk')
        plt.style.use("dark_background")

        metrics = metrics if metrics else self._get_scalar_metrics()
        for c in metrics:
            if c == 'epoch':
                continue
            try:
                self.plot_scalar(self.train_metrics[c], self.valid_metrics[c], best_epoch=self.best_epoch)
                print()
            except Exception as e:
                logger.warning(f"Failed to plot {c} due to the following error: {e}")

    @property
    def best_epoch(self) -> int:
        """Returns the best epoch for the given selection metric."""
        return self.selection_metric['epoch'].max()

    def get_best_epoch_summary(self):
        """Plots best epoch summary metrics."""

        with sns.axes_style("ticks"), sns.plotting_context('talk'), plt.style.context("dark_background"):
            best_epoch = self.best_epoch
            train_metrics, valid_metrics = self.train_metrics, self.valid_metrics
            print(colored(f'Best Epoch = {best_epoch}', 'green', attrs=['reverse', 'bold']))
            print()
            print(
                colored(
                    '----------------------------    Best Epoch Train Summary    ----------------------------',
                    'green', attrs=['reverse', 'bold']
                )
            )
            print()
            self.plot_single_epoch(train_metrics[train_metrics['epoch'] == best_epoch].iloc[0, :])
            print()
            print(
                colored(
                    '----------------------------    Best Epoch Valid Summary    ----------------------------',
                    'green', attrs=['reverse', 'bold']
                )
            )
            print()
            self.plot_single_epoch(valid_metrics[valid_metrics['epoch'] == best_epoch].iloc[0, :])

    def plot_single_epoch(self, epoch_data: pd.Series):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def plot_debug_images(self, mode: str, epochs: List[int] = None) -> None:
        """Plots all debug images saved during experiment run."""
        df = self.images_df
        df = df[df['mode'] == mode]

        if epochs:
            df = df[df['epoch'].isin(epochs)]

        for _, row in df.iterrows():
            plot_rgb_image(
                Path(row['image_path']),
                figsize=self._figsize,
                title=(
                    f'Epoch={row["epoch"]}-Batch={row["batch"]}-Image={row["image_index"]}_y_true ='
                    f' {row["y_true"]}_y_pred ={row["y_pred"]}'
                )
            )
            print()

    def show_config(self):
        with open(self._run_config_path) as f:
            config_data = json.load(f)
        print(json.dumps(config_data, indent=4, sort_keys=True))
