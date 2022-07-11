import numpy as np
from tqdm import tqdm


class EarlyStopping:
    """Early stops the training if the validation or train loss doesn't improve after a given patience (epochs)."""

    def __init__(self, patience: int = 7, delta: float = 1e-6, direction: str = 'min'):
        """
        Args:
            patience: How long to wait after last time validation loss improved.
                      If np.inf, then means no early stopping is applied
                      Default: 7
            delta: Minimum change in the monitored quantity to qualify as an improvement.
                   Default: 1e-6
            direction: min or max
        """
        assert direction in ('min', 'max'), direction
        self._patience = patience
        self._delta = delta

        self._early_stop = False
        self._counter = 0
        self._best_value = None
        self._direction = direction

    def adjust_direction(self, direction: str):
        """Adjusts direction."""
        assert direction in ('min', 'max'), direction
        assert self._counter == 0
        self._direction = direction

    @property
    def early_stop(self) -> bool:
        """Returns the boolean indicating if the early stopping should be done or not."""
        return self._early_stop

    def __call__(self, current_value: float, progress_bar: tqdm = None):
        """
        Args:
            current_value: current value
            progress_bar: the tqdm progress bar to write logs
        """
        if self._best_value is None:
            self._best_value = current_value

        else:
            if self._direction == 'min':
                check = self._best_value - current_value > self._delta  # current value is smaller from best value
            else:
                check = current_value - self._best_value > self._delta  # current value is smaller from best value

            if check:  # improvement
                self._best_value = current_value
                self._counter = 0
            else:
                self._counter += 1

                if progress_bar is not None and self._patience < np.inf:
                    progress_bar.write(f'          EarlyStopping counter: {self._counter} out of {self._patience}')

                if self._counter >= self._patience:
                    self._early_stop = True
                    if progress_bar is not None:
                        progress_bar.write(f'          EarlyStopping will be applied.')
