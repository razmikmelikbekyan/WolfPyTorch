import ast
from collections import defaultdict
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, Tuple, Mapping, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display import display
from clearml import Logger

from ..base_logger import BaseExperimentLogger
from ..base_visualizer import BaseExperimentsResultVisualizer
from ....logger import logger
from ....plotting_service.image import plot_rgb_image

__all__ = [
    'TLRegressionExperimentLogger', 'TLRegressionExperimentVisualizer'
]


class TLRegressionExperimentLogger(BaseExperimentLogger):
    """Class for storing and saving tile level regression experiments results."""

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
        """Plots regression images."""
        debug_image = np.transpose(debug_image, (0, 2, 3, 1))
        if images_dir:
            images_dir = Path(images_dir)

        for i in range(min(debug_image.shape[0], n_images_to_save)):
            try:
                image = np.clip(debug_image[i].squeeze().astype(np.float32), 0, 1)
                saving_path = images_dir.joinpath(f'{epoch_mode}_{epoch_number}_{batch_number}_{i}.png')
                info['image_index'] = i
                info['image_path'] = saving_path
                info['y_true'] = y_true[i]
                info['y_pred'] = y_pred[i]

                cls.save_image(image, str(saving_path), as_uint8=True)

                if clearml_logger is not None:
                    fig = plot_rgb_image(
                        image, figsize=cls.FIG_SIZE,
                        title=f'y_true={info["y_true"][0] :.2f}, y_pred={info["y_pred"][0] :.2f}',
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


class TLRegressionExperimentVisualizer(BaseExperimentsResultVisualizer):
    """Special class for view Regression Experiment results."""

    _figsize = TLRegressionExperimentLogger.FIG_SIZE

    @staticmethod
    def style_table(df: pd.DataFrame, axis: int, precision: int = 0):
        properties = {
            'font-size': '10pt',
            'background-color': 'white',
            'border-color': 'black',
            'border-style': 'solid',
            'border-width': '1px',
            'border-collapse': 'collapse',
            'width': '80px'
        }
        if precision == 0:
            str_format = "{:.0f}"
        elif precision == 1:
            str_format = "{:.1f}"
        elif precision == 2:
            str_format = "{:.2f}"
        elif precision == 3:
            str_format = "{:.3f}"
        elif precision == 4:
            str_format = "{:.4f}"
        else:
            raise ValueError

        return (df
                .style
                .set_properties(**properties)
                .background_gradient(cmap='OrRd', axis=axis)
                .format(str_format, subset=df.select_dtypes(include='number').columns)
                )

    @classmethod
    def plot_single_epoch(cls, epoch_data: pd.Series) -> Tuple[Mapping, pd.DataFrame]:
        """Plots Regression Epoch data."""
        epoch_data, regression_report = cls._parse_epoch_data(epoch_data)
        display(cls.style_table(regression_report, axis=0, precision=4))
        return epoch_data, regression_report

    @staticmethod
    def _parse_epoch_data(epoch_data: pd.Series) -> Tuple[Mapping, pd.DataFrame]:
        """Parses saved epoch data."""
        epoch_data = epoch_data.to_dict()
        regression_report = defaultdict(dict)
        for metric_name, metric_value in epoch_data.items():
            regression_report[metric_name] = metric_value

        regression_report = pd.DataFrame.from_dict(regression_report, orient="index").T
        regression_report.index = ['value']

        return epoch_data, regression_report

    def plot_debug_images(self, mode: str, epochs: List[int] = None) -> None:
        """Plots all debug images saved during experiment run."""
        df = self.images_df
        df = df[df['mode'] == mode]

        df['y_true'] = df['y_true'].apply(ast.literal_eval).apply(lambda x: x[0])
        df['y_pred'] = df['y_pred'].apply(ast.literal_eval).apply(lambda x: x[0])

        if epochs:
            df = df[df['epoch'].isin(epochs)]

        for _, row in df.iterrows():
            plot_rgb_image(
                Path(row['image_path']),
                figsize=self._figsize,
                title=(
                    f'Epoch={row["epoch"]}-Batch={row["batch"]}-Image={row["image_index"]}, y_true='
                    f' {row["y_true"] :.2f}, y_pred={row["y_pred"] :.2f}'
                ),
                brightness_factor=1.5,
            )
            print()
