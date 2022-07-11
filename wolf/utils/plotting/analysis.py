"""Module contains several functions used for tabular data analysis"""
from typing import Tuple, Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_countplot(df: pd.DataFrame,
                   x_column: str,
                   hue_column: str = None,
                   title: str = None,
                   figsize: Tuple[int, int] = (14, 8),
                   palette: Dict = None,
                   color: str or List[str] = "darkorange",
                   ylims: Tuple = None,
                   ticks_rotation: int = 90,
                   ticks_fontsize: int = 10,
                   annotation_rotation: int = 0,
                   order: bool or List = True,
                   output_path: str = None):
    """Plots count plot."""
    title = title if title else f'Count of {x_column}'

    plt.figure(figsize=figsize)
    plt.title(title, fontweight='bold', fontsize=16)

    if isinstance(order, bool) and order:
        order = df[x_column].value_counts().index
    elif isinstance(order, List) and order:
        order = order
    else:
        order = None

    splot = sns.countplot(x=x_column, hue=hue_column, data=df, palette=palette, color=color, order=order)
    for p in splot.patches:
        splot.annotate(
            f'{p.get_height():,.0f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center' if ticks_rotation == 0 else 'bottom',
            xytext=(0, 10),
            textcoords='offset points',
            fontweight='bold',
            rotation=annotation_rotation,
            fontsize=ticks_fontsize
        )
    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=ticks_fontsize, fontweight='bold')
    plt.xlabel(x_column, fontsize=12, fontweight='bold')
    plt.ylabel('count', fontsize=12, fontweight='bold')

    if ylims:
        plt.ylim(*ylims)

    sns.despine()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_aggregated_barplot(df: pd.DataFrame,
                            x_column: str,
                            y_column: str,
                            aggregation_type: str = None,
                            hue_column: str = None,
                            hue_dict: Dict = None,
                            color: str or List[str] = 'darkorange',
                            title: str = None,
                            figsize: Tuple[int, int] = (14, 8),
                            ylims: Tuple = None,
                            ticks_rotation: int = 90,
                            ticks_fontsize: int = 10,
                            annotation_precision: int = 0,
                            output_path: str = None):
    """Plots aggregated numerical column versus categories."""

    def hue_aggregation():
        df_agg = (
            df.groupby([x_column, hue_column]).agg({y_column: aggregation_type}).reset_index()
        )
        df_agg = pd.concat(
            [df_agg[[x_column, hue_column]], df_agg.loc[:, pd.IndexSlice[:, aggregation_type]]],
            axis=1
        )
        df_agg.columns = df_agg.columns.droplevel(1)
        return df_agg

    if hue_column:
        df_aggregated = hue_aggregation()
        title = title if title else f'{aggregation_type} {y_column} by {x_column}'
    elif aggregation_type:
        df_aggregated = df.groupby(x_column).agg({y_column: aggregation_type}).reset_index()
        title = title if title else f'{aggregation_type} {y_column} by {x_column}'
    else:
        df_aggregated = df
        title = title if title else f'{y_column} by {x_column}'

    plt.figure(figsize=figsize)
    plt.title(title, fontweight='bold', fontsize=16)
    splot = sns.barplot(
        x=x_column,
        y=y_column,
        hue=hue_column,
        data=df_aggregated,
        palette=hue_dict,
        color=color
    )
    sns.despine()

    for p in splot.patches:
        splot.annotate(
            f'{p.get_height():,.{annotation_precision}f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center' if ticks_rotation == 0 else 'bottom',
            xytext=(0, 10),
            textcoords='offset points',
            fontweight='bold',
            rotation=ticks_rotation,
            fontsize=ticks_fontsize
        )

    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=ticks_fontsize, fontweight='bold')
    plt.xlabel(x_column, fontsize=12, fontweight='bold')
    plt.ylabel(y_column, fontsize=12, fontweight='bold')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    if ylims:
        plt.ylim(*ylims)

    plt.tight_layout()
    plt.show()


def plot_lineplot(df: pd.DataFrame,
                  x_column: str,
                  y_column: str,
                  kind='line',
                  hue_column: str = None,
                  hue_dict: Dict = None,
                  color: str or List[str] = 'darkorange',
                  title: str = None,
                  figsize: Tuple[int, int] = (14, 8),
                  ticks_rotation: int = 90,
                  ticks_fontsize: int = 10,
                  grid: bool = True,
                  output_path: str = None,
                  ):
    title = title if title else f'{y_column} over {x_column}'

    plt.figure(figsize=figsize)
    plt.title(title, fontweight='bold', fontsize=16)
    if kind == 'line':
        sns.lineplot(
            x=x_column,
            y=y_column,
            hue=hue_column,
            data=df,
            palette=hue_dict,
            color=color,
        )
    elif kind == 'scatter':
        sns.scatterplot(
            x=x_column,
            y=y_column,
            hue=hue_column,
            data=df,
            palette=hue_dict,
            color=color,
        )
    elif kind == 'reg':
        sns.regplot(
            x=x_column,
            y=y_column,
            scatter=True,
            data=df,
            color=color,
            line_kws={'color': 'red'}
        )
    else:
        raise ValueError(f"Wrong kind={kind}.")

    sns.despine()

    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=ticks_fontsize, fontweight='bold')
    plt.xlabel(x_column, fontsize=12, fontweight='bold')
    plt.ylabel(y_column, fontsize=12, fontweight='bold')
    if grid:
        plt.grid(True, axis='both', linestyle='--', color='gray')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_normalised_barplot(df: pd.DataFrame,
                            x_column: str,
                            hue_column: str,
                            title: str = None,
                            figsize: Tuple[int, int] = (14, 8),
                            palette: Dict = None,
                            legend: bool = True,
                            ylims: Tuple = None,
                            ticks_rotation: int = 90,
                            ticks_fontsize: int = 10,
                            annotation_rotation: int = 90,
                            precision: int = 2,
                            order: List[str] = None,
                            output_path: str = None):
    """
    Plots percentage of every category from hue_column in division of categories from x_column.
    """
    title = title if title else f'percentage of {hue_column} in {x_column}'
    x, y, hue = x_column, "percentage", hue_column
    normalized_df = (df[hue_column]
                     .groupby(df[x])
                     .value_counts(normalize=True)
                     .rename(y)
                     .reset_index())
    normalized_df[y] *= 100

    plt.figure(figsize=figsize)
    plt.title(title, fontweight='bold', fontsize=16)
    ax = sns.barplot(x=x, y=y, hue=hue, data=normalized_df, palette=palette, order=order)
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), f'.{precision}f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center' if annotation_rotation == 0 else 'bottom',
            xytext=(0, 10),
            textcoords='offset points',
            fontweight='bold',
            rotation=annotation_rotation,
            fontsize=ticks_fontsize
        )
    sns.despine()

    if legend:
        plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    else:
        ax.get_legend().set_visible(False)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=ticks_fontsize, fontweight='bold')
    plt.xlabel(x, fontsize=13, fontweight='bold')
    plt.ylabel(f'{y} (%)', fontsize=13, fontweight='bold')

    if ylims:
        plt.ylim(*ylims)

    plt.tight_layout()
    plt.show()


def plot_normalised_count_plot(df: pd.DataFrame,
                               x_column: str,
                               title: str = None,
                               figsize: Tuple[int, int] = (14, 8),
                               palette: Dict = None,
                               ylims: Tuple = None,
                               ticks_rotation: int = 90,
                               ticks_fontsize: int = 10,
                               annotation_rotation: int = 0,
                               precision: int = 2,
                               output_path: str = None,
                               order: List = None):
    """
    Plots percentage of every category from hue_column in division of categories from x_column.
    """
    title = title if title else f'percentage of {x_column}'
    x, y = x_column, "percentage",
    normalized_df = pd.DataFrame(df[x].value_counts(normalize=True).rename('percentage').reset_index())
    normalized_df['percentage'] *= 100
    if order:
        normalized_df = normalized_df.set_index('index').loc[order].reset_index()

    plt.figure(figsize=figsize)
    plt.title(title, fontweight='bold', fontsize=16)
    ax = sns.barplot(x='index', y=y, data=normalized_df, palette=palette)
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), f'.{precision}f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center' if ticks_rotation == 0 else 'bottom',
            xytext=(0, 10),
            textcoords='offset points',
            fontweight='bold',
            rotation=annotation_rotation,
            fontsize=ticks_fontsize
        )
    sns.despine()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=ticks_fontsize, fontweight='bold')
    plt.xlabel(x, fontsize=12, fontweight='bold')
    plt.ylabel(y, fontsize=12, fontweight='bold')

    if ylims:
        plt.ylim(*ylims)

    plt.tight_layout()
    plt.show()


def plot_distplot(df: pd.DataFrame,
                  x_column: str,
                  hue_column: str = None,
                  norm_hist: bool = False,
                  hist: bool = False,
                  kde: bool = True,
                  legend: bool = True,
                  figsize: Tuple[int, int] = (10, 6),
                  title: str = None,
                  ticks_rotation: int = 90,
                  ticks_fontsize: int = 12,
                  output_path: str = None):
    """Plots distribution of given column."""
    plt.figure(figsize=figsize)
    if hue_column:
        if kde:
            if df[hue_column].dtype.name == 'category':
                values = df[hue_column].unique().sort_values()
            else:
                values = sorted(df[hue_column].unique())
            for x in values:
                sns.kdeplot(
                    df[df[hue_column] == x][x_column],
                    linewidth=3,
                    shade=True,
                    label=x,
                    hue_norm=norm_hist,
                )
        else:
            sns.histplot(
                data=df,
                x=x_column,
                hue=hue_column,
            )

    else:
        if kde:
            sns.kdeplot(
                df[x_column],
                linewidth=3,
                shade=True,
                hue_norm=norm_hist,
            )
        else:
            sns.histplot(
                data=df,
                x=x_column,
                hue=hue_column,
                hue_norm=norm_hist,
            )

    # Plot formatting
    if legend:
        plt.legend(prop={'size': 16})

    if title:
        plt.title(title, fontweight='bold')
    else:
        plt.title(f'Density Plot vs {hue_column}' if hue_column else 'Density Plot', fontweight='bold')
    plt.xlabel(x_column)
    plt.ylabel('Density')

    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=ticks_fontsize, fontweight='bold')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def plot_category_split_plot(df: pd.DataFrame,
                             figsize: Tuple[int, int] = (12, 6),
                             title: str = None,
                             x_label: str = None,
                             ticks_rotation: int = 90,
                             ticks_fontsize: int = 12,
                             legend: bool = True,
                             ylims: tuple = None,
                             output_path: str = None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    df.plot(ax=ax, kind='bar', stacked=True, rot=0,
            color=sns.color_palette('tab10', n_colors=10))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])
    ax.set_axisbelow(True)
    if legend:
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    if x_label:
        plt.xlabel(x_label, fontsize=ticks_fontsize, fontweight='bold')
    plt.ylabel('Percentage', fontsize=12, fontweight='bold')

    plt.grid(True, axis='y', linestyle='--', color='orange')
    if ylims:
        plt.ylim(*ylims)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    sns.despine()
    plt.show()


def plot_bar_plot(df: pd.DataFrame,
                  x_column: str,
                  y_column: str,
                  hue_column: str = None,
                  figsize: Tuple[int, int] = (12, 6),
                  title: str = None,
                  x_label: str = None,
                  ticks_rotation: int = 90,
                  ticks_fontsize: int = 12,
                  legend: bool = True,
                  ylims: tuple = None,
                  output_path: str = None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if title:
        plt.title(title, fontsize=14, fontweight='bold')

    sns.barplot(ax=ax, x=x_column, y=y_column, data=df, hue=hue_column)
    ax.set_axisbelow(True)
    if legend:
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    ax.yaxis.label.set_size(10)
    if x_label:
        plt.xlabel(x_label, fontsize=ticks_fontsize, fontweight='bold')

    plt.grid(True, axis='y', linestyle='--', color='orange')
    if ylims:
        plt.ylim(*ylims)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    sns.despine()
    plt.show()


def plot_boxplot(df: pd.DataFrame,
                 x_column: str,
                 y_column: str,
                 kind: str = 'box',
                 whis: float = 1.5,
                 hue_column: str = None,
                 color: str or List[str] = None,
                 title: str = None,
                 figsize: Tuple[int, int] = (14, 8),
                 palette: Dict = None,
                 ylims: Tuple = None,
                 ticks_rotation: int = 90,
                 ticks_fontsize: int = 10,
                 order: List[str] = None,
                 output_path: str = None):
    """Plots count plot."""
    plt.figure(figsize=figsize)

    if title:
        plt.title(title, fontweight='bold', fontsize=16)

    if kind == 'box':
        sns.boxplot(x=x_column, y=y_column, hue=hue_column, data=df,
                    palette=palette, color=color, order=order, whis=whis)
    elif kind == 'violin':
        sns.violinplot(x=x_column, y=y_column, hue=hue_column, data=df,
                       palette=palette, color=color, order=order, )
    elif kind == 'boxen':
        sns.boxenplot(x=x_column, y=y_column, hue=hue_column, data=df,
                      palette=palette, color=color, order=order)
    else:
        raise NotImplementedError

    plt.xticks(rotation=ticks_rotation, fontsize=ticks_fontsize, fontweight='bold')
    plt.yticks(fontsize=ticks_fontsize, fontweight='bold')
    plt.xlabel(x_column, fontsize=12, fontweight='bold')
    plt.ylabel(y_column, fontsize=12, fontweight='bold')

    if ylims:
        plt.ylim(*ylims)

    sns.despine()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_correlationplot(df: pd.DataFrame,
                         figsize: Tuple[int, int] = (12, 12),
                         annot: bool = True,
                         fontsize: int = 15,
                         title: str = None,
                         output_path: str = None
                         ):
    """Plots correlation matrix."""

    plt.figure(figsize=figsize)
    sns.heatmap(
        df,
        vmax=1.0, vmin=-1.0, linewidths=0.1,
        cmap="GnBu",
        fmt=".3f",
        annot=annot,
        annot_kws={"size": fontsize, "color": "white", "fontweight": "bold", },
        square=True
    )
    if title:
        plt.title(title, fontsize=18, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
