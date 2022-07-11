"""The module contains functions for plotting classification related stuff."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ["plot_classification_report", "plot_confusion_matrix"]


def plot_classification_report(classification_report: pd.DataFrame,
                               saving_path: str or Path = None,
                               return_figure: bool = False) -> Optional[plt.Figure]:
    """
    Plots classification report.
    Args:
        classification_report: the pandas dataframe representing the classification report
        saving_path: the path to save the figure
        return_figure: if True returns the Figure object

    Returns:
        matplotlib.pyplot.Figure object
    """

    with sns.axes_style("ticks"), sns.plotting_context('talk'), plt.style.context("dark_background"):
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(
            classification_report,
            annot=True,
            linewidths=0.2,
            cmap='RdYlGn',
            cbar=False,
            annot_kws={"fontweight": "bold"}
        )
        plt.title("Classification Report", size=18, weight='bold')
        plt.tight_layout()
        if saving_path:
            plt.savefig(saving_path, dpi=100)

        if return_figure:
            return fig
        else:
            plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          annotate_samples: bool = True,
                          saving_path: str or Path = None,
                          return_figure: bool = False) -> Optional[plt.Figure]:
    """
    Generate matrix plot of confusion matrix with pretty annotations.

    Args:
        cm: the array representing the confusion matrix, must be a square array
        class_names: the class names of the confusion matrix, the order must be the same as in confusion matrix
        annotate_samples: if True will show numbers on the plot
        saving_path: the path to save the figure
        return_figure: if True returns the Figure object

    Returns:
        matplotlib.pyplot.Figure object
    """
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if not annotate_samples:
                annot[i, j] = f'{0 if np.isnan(p) else p :.1f}%'
            else:
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = f'{0 if np.isnan(p) else p :.1f}%\n{int(c)}/\n{int(s)}'
                else:
                    annot[i, j] = f'{0 if np.isnan(p) else p :.1f}%\n{int(c)}'

    cm = pd.DataFrame(cm_perc, index=class_names, columns=class_names)
    cm = cm.fillna(0.)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    with sns.axes_style("ticks"), sns.plotting_context('talk'), plt.style.context("dark_background"):
        fig = plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=annot, fmt='', mask=~off_diag_mask, cmap='PRGn', cbar=False, linecolor='black',
                    linewidths=1., annot_kws={"fontweight": "bold"})
        sns.heatmap(cm, annot=annot, fmt='', mask=off_diag_mask, cmap='Reds', cbar=False, linecolor='black',
                    linewidths=1., annot_kws={"fontweight": "bold", "fontsize": 14})
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.title('Confusion Matrix', size=18, weight='bold')
        plt.tight_layout()
        if saving_path:
            plt.savefig(saving_path, dpi=100)

        if return_figure:
            return fig
        else:
            plt.show()
