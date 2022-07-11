"""The module contains functions for plotting classification related stuff."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = ["plot_prediction_vs_actual", "plot_residuals", "plot_mixtures", "plot_quantiles"]


def plot_prediction_vs_actual(df: pd.DataFrame):
    """Plots prediction vs actual scatter plot."""

    all_items = set(df['train_valid_test'])
    columns = [x for x in ['train', 'valid', 'test'] if x in all_items]
    color_mapper = {'train': 'green', 'valid': 'orange', 'test': 'red'}
    n = len(columns)

    fig, axes = plt.subplots(1, n, figsize=(n * 10, 8))
    for i in range(n):
        c, ax = columns[i], (axes[i] if n > 1 else axes)
        tmp = df[df['train_valid_test'] == c]
        min_value = min(tmp['y_true'].min(), tmp['y_pred'].min())
        max_value = max(tmp['y_true'].max(), tmp['y_pred'].max())
        sns.scatterplot(
            x='y_pred',
            y='y_true',
            data=tmp,
            color=color_mapper[c],
            ax=ax,
            alpha=0.3
        )
        ax.plot([min_value, max_value], [min_value, max_value], color='blue', linewidth=1, alpha=0.5)
        ax.grid(True, axis='both', linestyle='--', color='gray')
        ax.set_xlim(min_value - min_value * 0.1, max_value + max_value * 0.1)
        ax.set_ylim(min_value - min_value * 0.1, max_value + max_value * 0.1)
        ax.xaxis.set_tick_params(labelbottom=True, rotation=0, size=14)
        ax.yaxis.set_tick_params(size=14)
        ax.set_xlabel('y_pred', fontsize=14, fontweight='bold')
        ax.set_ylabel('y_true', fontsize=14, fontweight='bold')
        ax.set_title(f'{c}', fontsize=16, fontweight='bold')

    plt.suptitle(f'Prediction vs Actual', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_residuals(df: pd.DataFrame):
    """Plots prediction vs actual scatter plot."""

    all_items = set(df['train_valid_test'])
    columns = [x for x in ['train', 'valid', 'test'] if x in all_items]
    color_mapper = {'train': 'green', 'valid': 'orange', 'test': 'red'}
    n = len(columns)

    fig, axes = plt.subplots(1, n, figsize=(n * 10, 8))
    for i in range(n):
        c, ax = columns[i], (axes[i] if n > 1 else axes)
        tmp = df[df['train_valid_test'] == c]
        sns.scatterplot(
            x='y_pred',
            y='residuals',
            data=tmp,
            color=color_mapper[c],
            ax=ax,
            alpha=0.3
        )
        ax.xaxis.set_tick_params(labelbottom=True, rotation=0, size=12)
        ax.yaxis.set_tick_params(size=14)
        ax.set_xlabel('y_pred', fontsize=14, fontweight='bold')
        ax.set_ylabel('residuals', fontsize=14, fontweight='bold')
        ax.set_title(f'{c}', fontsize=16, fontweight='bold')
        ax.axhline(y=0, color='blue', alpha=0.3, linewidth=1)
        ax.grid(True, axis='both', linestyle='--', color='gray')

    plt.suptitle(f'Prediction vs Residuals', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_quantiles(df: pd.DataFrame, n_samples: int = 500):
    """Plots prediction vs actual scatter plot."""
    quantiles = sorted(
        [c for c in df.columns if isinstance(c, str) and c.startswith('q_')],
        key=lambda x: int(x.split('_')[-1])
    )

    if not quantiles:
        return
    n_quantiles = len(quantiles)
    assert n_quantiles % 2 == 0

    all_items = set(df['train_valid_test'])
    columns = [x for x in ['train', 'valid', 'test'] if x in all_items]
    n = len(columns)

    fig, axes = plt.subplots(1, n, figsize=(n * 10, 8))
    for i in range(n):
        c, ax = columns[i], (axes[i] if n > 1 else axes)
        tmp = df[df['train_valid_test'] == c].copy()
        tmp = tmp.sample(n=min(len(tmp), n_samples), random_state=32)
        tmp = tmp.sort_values(by='y_true')

        ax.plot(range(len(tmp)), tmp[quantiles[0]].values, "^", linewidth=1, label=quantiles[0], alpha=0.5)
        ax.plot(range(len(tmp)), tmp['y_true'].values, '.', color='green', alpha=0.3, label='y_true')
        ax.plot(range(len(tmp)), tmp[quantiles[-1]].values, "v", linewidth=1, label=quantiles[-1], alpha=0.5)

        ax.xaxis.set_tick_params(labelbottom=True, rotation=0, size=14)
        ax.yaxis.set_tick_params(size=14)
        ax.set_xlabel('samples', fontsize=14, fontweight='bold')
        ax.set_ylabel('target and predictions', fontsize=14, fontweight='bold')
        ax.set_title(f'{c}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left')

    plt.suptitle(f'Quantiles vs Actual', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


# TODO: adjust
def plot_mixtures(df: pd.DataFrame, n_samples: int = 50, sigma_factor=1):
    """Plots prediction vs actual scatter plot."""
    all_items = set(df['train_valid_test'])
    columns = [x for x in ['train', 'valid', 'test'] if x in all_items]
    n = len(columns)

    fig, axes = plt.subplots(1, n, figsize=(n * 10, 8))
    for i in range(n):
        c, ax = columns[i], (axes[i] if n > 1 else axes)
        tmp = df[df['train_valid_test'] == c].copy()
        tmp = tmp.sample(n=min(len(tmp), n_samples), random_state=32)
        tmp = tmp.sort_values(by='y_true')

        ax.plot(range(len(tmp)), tmp['y_true'].values - sigma_factor * tmp['mixture_sigmas'].values, "^", linewidth=1,
                label=f'lower_bound(mean - {sigma_factor} * sigma)', alpha=0.5)
        ax.plot(range(len(tmp)), tmp['y_true'].values, '.', color='green', alpha=0.3, label='y_true')
        ax.plot(range(len(tmp)), tmp['y_true'].values + sigma_factor * tmp['mixture_sigmas'].values, "v", linewidth=1,
                label=f'upper_bound(mean + {sigma_factor} * sigma)', alpha=0.5)

        ax.xaxis.set_tick_params(labelbottom=True, rotation=0, size=14)
        ax.yaxis.set_tick_params(size=14)
        ax.set_xlabel('samples', fontsize=14, fontweight='bold')
        ax.set_ylabel('target and predictions', fontsize=14, fontweight='bold')
        ax.set_title(f'{c}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left')

    plt.suptitle(f'Predicted Mean with its lower and higher boundary', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
