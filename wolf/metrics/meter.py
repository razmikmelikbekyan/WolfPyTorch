from typing import Any

import numpy as np

__all__ = ['Meter', 'AverageValueMeter']


class Meter:
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Resets the meter to default settings."""
        raise NotImplementedError

    def add(self, value: Any):
        """Log a new value to the meter.
        Args:
            value: Next result to include.
        """
        raise NotImplementedError


class AverageValueMeter(Meter):
    """Defines the average meter."""

    def __init__(self):
        super().__init__()

        self._n = 0
        self._old_mean = 0
        self._new_mean = np.nan
        self._old_sum_std = 0
        self._new_sum_std = np.nan

        self.reset()

    def add(self, x: float):
        self._n += 1.

        if self._n == 1:
            self._old_mean = self._new_mean = x
            self._old_sum_std, self._new_sum_std = 0, np.nan
        else:
            self._new_mean = self._old_mean + (x - self._old_mean) / self._n
            self._new_sum_std = self._old_sum_std + (x - self._old_mean) * (x - self._new_mean)

            self._old_mean = self._new_mean
            self._old_sum_std = self._new_sum_std

    @property
    def mean(self) -> float:
        """Returns the mean value."""
        return self._new_mean

    @property
    def std(self) -> float:
        """Returns the std value."""
        return np.sqrt(self._new_sum_std / (self._n - 1)) if self._n > 1 else np.nan

    def reset(self):
        self._n = 0
        self._old_mean = 0
        self._new_mean = np.nan
        self._old_sum_std = 0
        self._new_sum_std = np.nan
