from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = ["plot_tabular_data"]


def plot_tabular_data(table_data: pd.DataFrame,
                      title: str,
                      w: float = 5,
                      h: float = 3,
                      saving_path: str or Path = None,
                      return_figure: bool = False) -> Optional[plt.Figure]:
    """
    Plots tabular data using matploltlib.pyplot.table object.
    Args:
        table_data: the pandas dataframe representing the data to plot
        title: the title of the data to plot
        w: width in inches for each column in the data
        h: the height of each 5 rows in the data
        saving_path: the path to save the figure
        return_figure: if True returns the Figure object

    Returns:
        matplotlib.pyplot.Figure object
    """
    with sns.axes_style("ticks"), sns.plotting_context('talk'), plt.style.context("dark_background"):
        table_data = table_data.copy()
        w = max(int((table_data.shape[1] / 1) * w), w)  # for each column 5 inch
        h = max(int((table_data.shape[0] / 5) * h), h)  # for each 5 rows 3 inch

        fig = plt.figure(figsize=(w, h), linewidth=2, tight_layout={'pad': 0.6})
        table = plt.table(
            cellText=table_data.applymap(lambda x: f'{x: .4f}').values.tolist(),
            rowLabels=table_data.index,
            colLabels=table_data.columns,
            rowColours=['w'] * table_data.shape[0],
            colColours=['w'] * table_data.shape[1],
            fontsize=20,
            colWidths=[0.3] * table_data.shape[1],
            loc='center'
        )

        for key, cell in table.get_celld().items():
            # adjust format for only header col and row to help with space
            # issues col header on 0, row header on -1.
            cell.get_text().set_fontsize(20)
            if key[0] == 0 or key[1] == -1:
                cell.get_text().set_color('blue')
            else:
                cell.get_text().set_color('red')

        table.scale(3, 3)

        plt.axis('off')
        plt.suptitle(title, color='blue', size=16, weight='bold')
        plt.draw()
        if saving_path:
            plt.savefig(saving_path, dpi=100)

        if return_figure:
            return fig
        else:
            plt.show()
