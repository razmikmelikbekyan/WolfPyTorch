from typing import Tuple, Union, Dict

import numpy as np
import torch
import torch.nn as nn


def split_mdn_output(n_components: int, mdn_output: Union[torch.Tensor, np.ndarray]) -> Tuple:
    """
    Splits the MDN output into pi, mean and sigma.
    Args:
        n_components: the number of MDN components
        mdn_output: the output of MDN Wrapper
    Returns:
        pi, mean, sigma
    """
    assert mdn_output.shape[1] == n_components * 3, "MDN Output 2nd dim must have size 3 * components"
    pi_start, mean_start, sigma_start = 0, n_components, 2 * n_components
    pi_end, mean_end, sigma_end = mean_start, sigma_start, 3 * n_components
    return (
        mdn_output[:, pi_start:pi_end, ...],  # pi
        mdn_output[:, mean_start:mean_end, ...],  # mean
        mdn_output[:, sigma_start:sigma_end, ...]  # sigma
    )


class PLMDNWrapper(nn.Module):
    """Pixel Level Mixture density network wrapper for Segmentation Models."""

    def __init__(self, mixture_components: int = 1):
        """
        Args:
            mixture_components: the number of components
        """
        super(PLMDNWrapper, self).__init__()
        self._mixture_components = mixture_components
        self._mdn_head: nn.Module = None

    @property
    def mixture_components(self) -> int:
        """Returns the number of components"""
        return self._mixture_components

    def wrap_model(self, segmentation_model: nn.Module) -> nn.Module:
        """Adjusts a segmentation model to have MDN head."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BottleneckMDNWrapper(PLMDNWrapper):
    """Bottleneck raising Mixture density network wrapper for Segmentation Models."""

    def wrap_model(self, segmentation_model: nn.Module) -> nn.Module:
        """Adjusts a segmentation model to have MDN head."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
