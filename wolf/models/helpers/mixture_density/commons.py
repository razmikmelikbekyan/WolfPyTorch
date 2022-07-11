from typing import Tuple, Union

import numpy as np
import torch


def join_mdn_output(pi_output: torch.Tensor, mean_output: torch.Tensor, sigma_output: torch.Tensor) -> torch.Tensor:
    """
    Joins the MDN outputs into one tensor.
    Args:
        pi_output: the output from PI layer
        mean_output: the output of MEAN layer
        sigma_output: the output of SIGMA layer
    Returns:
        combined MDN output
    """
    assert pi_output.shape == mean_output.shape == sigma_output.shape, "PI, MEAN and SIGMA must have the sampe shape"
    return torch.cat([pi_output, mean_output, sigma_output], dim=1)


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
        mdn_output[:, pi_start:pi_end],  # pi
        mdn_output[:, mean_start:mean_end],  # mean
        mdn_output[:, sigma_start:sigma_end]  # sigma
    )
