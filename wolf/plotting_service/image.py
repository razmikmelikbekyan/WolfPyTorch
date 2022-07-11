"""The module contains functions for plotting images."""

from pathlib import Path
from typing import Tuple, Union, Optional, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_rgb_image", "plot_multiple_images"]


def plot_rgb_image(image: Union[Path, str, np.ndarray],
                   title: str,
                   figsize: Tuple[int, int],
                   brightness_factor: float = None,
                   return_figure: bool = False
                   ) -> Optional[plt.Figure]:
    """
    Plots the RGB image.
    Args:
        image: the image array or the path to read the image
        title: the title of the image to plot
        figsize: the tuple of ints specifying the figure size
        brightness_factor: in some cases we may want to increase the brightness of the image, so if you specify this
                           factor the plotted image will be multiplied by this factor
        return_figure: if True returns the Figure object

    Returns:
        matplotlib.pyplot.Figure object
    """
    if not isinstance(image, np.ndarray):
        image = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if brightness_factor:
        image = image * brightness_factor
        image = image / image.max()

    with plt.style.context('default'):
        fig = plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.title(title, size=12)
        plt.axis('off')
        if return_figure:
            return fig
        else:
            plt.show()


def plot_multiple_images(images: List[np.ndarray],
                         titles: List[str],
                         figsize: Tuple[int, int],
                         return_figure: bool = False
                         ) -> Optional[plt.Figure]:
    """
    Plots multiple images in one row, the the resulting plot has 1 row and N = len(images) columns.
    Args:
        images: list of image arrays to be plotted, each array must be 2D or 3D
        titles: the list of strings representing the titles of each image
        figsize: the tuple of ints specifying the figure size
        return_figure: if True returns the Figure object

    Returns:
        matplotlib.pyplot.Figure object
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontweight="bold", size=10)
        axes[i].axis('off')

    plt.tight_layout()
    if return_figure:
        return fig
    else:
        plt.show()
