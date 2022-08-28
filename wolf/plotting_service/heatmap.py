"""The module contains functions for plotting heatmaps.

For more details please check here:
https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""

from pathlib import Path
from typing import Union, Tuple, Optional

import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

__all__ = ["plot_2_heatmaps"]


def plot_2_heatmaps(image_1: Union[Path, str, np.ndarray],
                    image_2: Union[Path, str, np.ndarray],
                    title_1: str,
                    title_2: str,
                    figsize: Tuple[int, int],
                    min_value_to_filter: float = None,
                    max_value_to_filter: float = None,
                    return_figure: bool = False
                    ) -> Optional[plt.Figure]:
    """
    Plots 2 heatmap images on the same row.
    It is useful function in case you want to visualize pixel-wise regression results: actual vs predicion.

    Args:
        image_1: the first image array (must be a square array) or path to it
        image_2: the second image array (must be a square array) or path to it
        title_1: the title of the first image to plot
        title_2: the title of the second image to plot
        figsize: the tuple of ints specifying the figure size
        min_value_to_filter: if given all the values in the image which are less than this value will be replaced by nan
        max_value_to_filter: if given all the values in the image which are մօռե than this value will be replaced by nan
        return_figure: if True returns the Figure object

    Returns:
        matplotlib.pyplot.Figure object
    """
    if not isinstance(image_1, np.ndarray):
        image_1 = cv2.imread(str(image_1), cv2.IMREAD_UNCHANGED)

    if not isinstance(image_2, np.ndarray):
        image_2 = cv2.imread(str(image_2), cv2.IMREAD_UNCHANGED)

    filtered_image_1, filtered_image_2 = image_1.copy(), image_2.copy()
    if min_value_to_filter is not None:
        filtered_image_1[filtered_image_1 < min_value_to_filter] = np.nan
        filtered_image_2[filtered_image_2 < min_value_to_filter] = np.nan
    if max_value_to_filter is not None:
        filtered_image_1[filtered_image_1 > max_value_to_filter] = np.nan
        filtered_image_2[filtered_image_2 > max_value_to_filter] = np.nan

    v_min = min(np.nanmin(filtered_image_1), np.nanmin(filtered_image_2))
    v_max = max(np.nanmax(filtered_image_1), np.nanmax(filtered_image_2))
    mean_1, mean_2 = np.nanmean(filtered_image_1), np.nanmean(filtered_image_2)

    with plt.style.context('default'):
        fig, axes = plt.subplots(
            ncols=3,
            figsize=figsize,
            gridspec_kw=dict(width_ratios=[1, 1, 0.08])
        )
        sns.heatmap(image_1, annot=False, ax=axes[0], cbar=False, vmin=v_min, vmax=v_max)
        axes[0].set_title(f'{title_1}_mean={mean_1 :.2f}')
        axes[0].axis('off')
        sns.heatmap(image_2, annot=False, ax=axes[1], cbar=False, vmin=v_min, vmax=v_max)
        axes[1].set_title(f'{title_2}_mean={mean_2 :.2f}')
        axes[1].axis('off')
        fig.colorbar(axes[1].collections[0], cax=axes[2])
        plt.tight_layout()
        if return_figure:
            return fig
        else:
            plt.show()
