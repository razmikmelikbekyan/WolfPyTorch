import itertools
from collections import defaultdict
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, Optional
from typing import Tuple, Mapping

import numpy as np
import pandas as pd
from clearml import Logger
from matplotlib import pyplot as plt

from yield_forecasting.utils.logger import logger
from ..base_logger import BaseExperimentLogger
from ..base_visualizer import BaseExperimentsResultVisualizer
from ....plotting_service.classification import plot_classification_report, plot_confusion_matrix
from ....plotting_service.image import plot_rgb_image
from ....plotting_service.tabular import plot_tabular_data

__all__ = [
    'TLClassificationExperimentLogger', 'TLClassificationEpochVisualizer'
]


class TLClassificationExperimentLogger(BaseExperimentLogger):
    """Class for storing and saving classification (binary and multiclass) experiments results."""

    FIG_SIZE = (10, 10)

    @classmethod
    def save_epoch_images(cls,
                          debug_image: np.ndarray,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          info: Dict,
                          mp_manager: Manager,
                          epoch_mode: str = None,
                          epoch_number: int = None,
                          batch_number: int = None,
                          images_dir: str = None,
                          clearml_logger: Optional[Logger] = None,
                          n_images_to_save: int = 10):
        """Plots classification images."""
        debug_image = np.transpose(debug_image, (0, 2, 3, 1))

        if images_dir:
            images_dir = Path(images_dir)

        for i in range(min(debug_image.shape[0], n_images_to_save)):
            try:
                image = np.clip(debug_image[i].squeeze().astype(np.float32), 0, 1)
                y_true = y_true[i]  # true class
                y_pred = y_pred[i].squeeze().astype(np.float32)  # class probabilities

                saving_path = images_dir.joinpath(f'{epoch_mode}_{epoch_number}_{batch_number}_{i}.png')
                info['image_index'] = i
                info['image_path'] = saving_path
                info['y_true'] = y_true
                info['y_pred'] = y_pred
                cls.save_image(image, str(saving_path), as_uint8=True)

                if clearml_logger is not None:
                    fig = plot_rgb_image(
                        image, figsize=cls.FIG_SIZE,
                        title=f"TrueClass={y_true} - PredictedClass={np.argmax(y_pred)}",
                        brightness_factor=1.5,
                        return_figure=True,
                    )
                    clearml_logger.report_matplotlib_figure(
                        title=f'{epoch_mode}/RGB-TL_COMPARISON',
                        series=f'batch={batch_number}_image={i}',
                        figure=fig,
                        iteration=epoch_number,
                        report_image=True,
                        report_interactive=False,
                    )
                    plt.close(fig=fig)
                mp_manager.append(info)
            except Exception as e:
                logger.warning(f'Plotting failed due to error: {e}', stack_info=True, exc_info=True)


class TLClassificationEpochVisualizer(BaseExperimentsResultVisualizer):
    """Special class for view Classification Epoch results."""

    _figsize = TLClassificationExperimentLogger.FIG_SIZE

    @classmethod
    def plot_single_epoch(cls,
                          epoch_data: pd.Series,
                          output_folder: str or Path = None) -> Tuple[Mapping, pd.DataFrame, pd.DataFrame]:
        """Plots Classification Epoch data."""
        if output_folder is not None:
            output_folder = Path(output_folder) if isinstance(output_folder, str) else output_folder

        epoch_data, total_metrics, classification_report = cls._parse_epoch_data(epoch_data)
        plot_tabular_data(
            total_metrics, 'TotalMetrics',
            w=5, h=3,
            saving_path=output_folder.joinpath('TotalMetrics.png') if output_folder else None,
        )
        print()
        plot_classification_report(
            classification_report,
            saving_path=output_folder.joinpath('ClassificationReport.png') if output_folder else None
        )
        print()
        cm = epoch_data.get('confusion_matrix')
        if cm is not None:
            plot_confusion_matrix(
                np.array(cm), epoch_data['class_names'],
                saving_path=output_folder.joinpath('ConfusionMatrix.png') if output_folder else None
            )
            print()

        return epoch_data, total_metrics, classification_report

    @staticmethod
    def _parse_epoch_data(epoch_data: pd.Series) -> Tuple[Mapping, pd.DataFrame, pd.DataFrame]:
        """Parses saved epoch data."""
        average_types = ('by_class', 'micro', 'macro', 'weighted')
        epoch_data = epoch_data.to_dict()

        classification_report = defaultdict(dict)
        for name, value in epoch_data.items():
            for x, y in itertools.product(('recall', 'precision', 'f1'), average_types):
                if (x in name) and (y in name):
                    if y == 'by_class':
                        assert isinstance(value, dict), value
                        for class_name, class_metric_value in value.items():
                            classification_report[x][class_name] = class_metric_value
                    else:
                        assert isinstance(value, (float, int))
                        classification_report[x][y] = value
                    break
        classification_report = pd.DataFrame(classification_report)

        total_metrics = {}
        table_metrics = ('accuracy', 'balanced_accuracy', 'top_k_accuracy_score', 'roc_auc_score')
        for name, value in epoch_data.items():
            for x in table_metrics:
                if x in name:
                    total_metrics[name] = value
                    break

        total_metrics = pd.DataFrame(pd.Series(total_metrics), columns=['metric_value'])

        return epoch_data, total_metrics, classification_report
