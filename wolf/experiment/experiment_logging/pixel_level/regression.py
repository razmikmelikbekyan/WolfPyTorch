from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Logger

from ..base_logger import BaseExperimentLogger
from ..tile_level.regression import TLRegressionExperimentVisualizer
from ....logger import logger
from ....plotting_service.heatmap import plot_2_heatmaps
from ....plotting_service.image import plot_rgb_image

__all__ = [
    'PLRegressionExperimentLogger', 'PLRegressionExperimentVisualizer'
]


class PLRegressionExperimentLogger(BaseExperimentLogger):
    """Class for storing and saving Pixel Level regression experiments results."""

    FIG_SIZE = (12, 12)

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
                          images_dir: str or Path = None,
                          clearml_logger: Optional[Logger] = None,
                          n_images_to_save: int = 10):
        debug_image = np.transpose(debug_image, (0, 2, 3, 1))

        if images_dir:
            images_dir = Path(images_dir)

        for i in range(min(debug_image.shape[0], n_images_to_save)):
            try:
                image = np.clip(debug_image[i].squeeze().astype(np.float32), 0, 1)
                input_saving_path = images_dir.joinpath(f'{epoch_mode}_{epoch_number}_{batch_number}_{i}.png')
                cls.save_image(image, str(input_saving_path), as_uint8=True)

                ground_truth = y_true[i].squeeze().astype(np.float32)
                y_true_saving_path = images_dir.joinpath(f'{epoch_mode}_y_true_{epoch_number}_{batch_number}_{i}.tif')
                cls.save_image(ground_truth, y_true_saving_path, as_uint8=False)

                prediction = y_pred[i].squeeze().astype(np.float32)
                y_pred_saving_path = images_dir.joinpath(f'{epoch_mode}_y_pred_{epoch_number}_{batch_number}_{i}.tif')
                cls.save_image(prediction, y_pred_saving_path, as_uint8=False)

                info['image_index'] = i
                info['image_path'] = input_saving_path
                info['y_pred_image_path'] = y_pred_saving_path
                info['y_true_image_path'] = y_true_saving_path

                if clearml_logger is not None:
                    fig = plot_rgb_image(
                        image,
                        figsize=cls.FIG_SIZE,
                        title=f'RGB',
                        brightness_factor=1.5,
                        return_figure=True,
                    )
                    clearml_logger.report_matplotlib_figure(
                        title=f'{epoch_mode}/RGB',
                        series=f'batch={batch_number}_image={i}',
                        figure=fig,
                        iteration=epoch_number,
                        report_image=True,
                        report_interactive=False,
                    )
                    plt.close(fig=fig)

                    fig = plot_2_heatmaps(
                        ground_truth,
                        y_pred_saving_path,
                        title_1=f'y_true',
                        title_2=f'y_pred',
                        figsize=(cls.FIG_SIZE[0] * 2, cls.FIG_SIZE[1]),
                        min_value_to_filter=None,
                        max_value_to_filter=None,
                        return_figure=True
                    )
                    clearml_logger.report_matplotlib_figure(
                        title=f'{epoch_mode}/PL_COMPARISON',
                        series=f'batch={batch_number}_image={i}',
                        figure=fig,
                        iteration=epoch_number,
                        report_image=True,
                        report_interactive=False,
                    )
                    plt.close(fig=fig)
                    plt.cla()

                mp_manager.append(info)
            except Exception as e:
                logger.warning(f'Plotting failed due to error: {e}', stack_info=True, exc_info=True)


class PLRegressionExperimentVisualizer(TLRegressionExperimentVisualizer):
    """Special class for view Regression Segmentation Experiment results."""

    _figsize = PLRegressionExperimentLogger.FIG_SIZE

    @property
    def images_df(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self._images_csv_path, index_col=0).copy()
        except FileNotFoundError:
            logger.warning("Image data is not yet finished saving, trying to recover from saved images.")
            try:
                images = list(self._images_dir.iterdir())

                input_images = [x for x in images if 'pred' not in x.stem and 'true' not in x.stem]
                y_true_real = [x for x in images if 'true' in x.stem]
                y_pred_real = [x for x in images if 'pred' in x.stem]

                y_true_dummy = [Path(x.parent.joinpath(str(x.name).replace('_y_true_', '_'))) for x in y_true_real]
                y_pred_dummy = [Path(x.parent.joinpath(str(x.name).replace('_y_pred_', '_'))) for x in y_pred_real]

                input_images_df = self._image_paths_to_dataframe(input_images)
                y_true_images_df = self._image_paths_to_dataframe(y_true_dummy, real_paths=y_true_real)
                y_pred_images_df = self._image_paths_to_dataframe(y_pred_dummy, real_paths=y_pred_real)

                y_true_images_df = y_true_images_df.rename(columns={'image_path': 'y_true_image_path'})
                y_pred_images_df = y_pred_images_df.rename(columns={'image_path': 'y_pred_image_path'})

                input_images_df = pd.merge(
                    input_images_df, y_true_images_df,
                    how='left', on=['mode', 'epoch', 'batch', 'image_index']
                )
                input_images_df = pd.merge(
                    input_images_df, y_pred_images_df,
                    how='left', on=['mode', 'epoch', 'batch', 'image_index']
                )

            except Exception as e:
                logger.info(f"Failed to recover image data from saved images, due to error: {e}")
                logger.warning("Image data is not yet finished saving, please try after training is finished.")
            else:
                return input_images_df

    def plot_debug_images(self,
                          mode: str, epochs: List[int] = None,
                          min_value_to_filter: float = None,
                          max_value_to_filter: float = None,
                          brightness_factor: float = 1,
                          ) -> None:
        """Plots all debug images saved during experiment run."""
        df = self.images_df
        df = df[df['mode'] == mode]

        if epochs:
            df = df[df['epoch'].isin(epochs)]

        for _, row in df.iterrows():
            plot_rgb_image(
                Path(row['image_path']),
                figsize=self._figsize,
                title=f'Epoch={row["epoch"]}-Batch={row["batch"]}-Image={row["image_index"]}-input',
                brightness_factor=brightness_factor
            )

            plot_2_heatmaps(
                Path(row['y_true_image_path']),
                Path(row['y_pred_image_path']),
                title_1=f'Epoch={row["epoch"]}-Batch={row["batch"]}-Image={row["image_index"]}-y_true',
                title_2=f'Epoch={row["epoch"]}-Batch={row["batch"]}-Image={row["image_index"]}-y_pred',
                figsize=(self._figsize[0] * 2, self._figsize[1]),
                min_value_to_filter=min_value_to_filter,
                max_value_to_filter=max_value_to_filter
            )
            print()
