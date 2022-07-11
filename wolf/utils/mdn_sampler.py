from typing import Tuple

import numpy as np
import torch
from torch.distributions import OneHotCategorical, Normal


def sum_multiple_gaussians(pi: torch.Tensor, mean: torch.Tensor,
                           sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A generalized formula from
    https://stats.stackexchange.com/questions/205126/standard-deviation-for-weighted-sum-of-normal-distributions
     """
    combined_mean = (mean * pi).sum(axis=1)
    combined_sigma = (pi * (np.square(sigma) + np.square(mean))).sum(axis=1)
    combined_sigma -= np.square(combined_mean)
    combined_sigma = np.sqrt(combined_sigma)
    return combined_mean, combined_sigma


class MDNSampler:
    """Special Class for Sampling from Gaussian Mixture Density Network Outputs."""

    # the most overhead is used for transferring tensors to cuda, so will do it only once
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    normal_dist = Normal(torch.tensor(0.).to(device=device), torch.tensor(1.).to(device=device))

    @classmethod
    def categorical_pl_sample(cls, weights: torch.Tensor, mask: torch.Tensor, random_seed: int) -> torch.Tensor:
        """Samples from pixel level categorical distribution."""
        weights = torch.masked_fill(weights, ~mask, 1. / weights.shape[0])
        categorical_dist = OneHotCategorical(probs=torch.transpose(weights, 0, 2))
        torch.manual_seed(random_seed)
        return torch.transpose(categorical_dist.sample(), 2, 0)

    @classmethod
    def categorical_tl_sample(cls, weights: torch.Tensor, random_seed: int) -> torch.Tensor:
        """Samples from tile level categorical distribution."""
        categorical_dist = OneHotCategorical(probs=weights)
        torch.manual_seed(random_seed)
        return categorical_dist.sample()

    @classmethod
    def normal_sample(cls, sample_shape: Tuple, random_seed) -> torch.Tensor:
        """Samples from standard normal distribution. """
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # normal = Normal(torch.tensor(0.).to(device=device), torch.tensor(1.).to(device=device))
        torch.manual_seed(random_seed)
        standard_sample = cls.normal_dist.sample(sample_shape)
        return standard_sample.cpu() if cls.device != 'cpu' else standard_sample

    @classmethod
    def pl_sample(cls, means: np.ndarray, sigmas: np.ndarray, pi: np.ndarray, mask: np.ndarray,
                  n_samples: int = 1000, random_seed: int = 42):
        """
        Pixel level sampling with Torch over large arrays.

        Args:
            means: [mixture_components, dim, dim] array
            sigmas: [mixture_components, dim, dim] array
            pi: [mixture_components, dim, dim] array
            mask: [dim, dim] array that 0 means that should be ignored
            random_seed: seed for random sampling
            n_samples: the number of sampling iterations
        Returns:
            the sampled values array with the shape [n_samples, dim, dim]
        """
        assert means.shape == sigmas.shape == pi.shape, (means.shape, sigmas.shape, pi.shape)
        assert means.ndim == sigmas.ndim == pi.ndim == 3, (means.shape, sigmas.shape, pi.shape)
        assert means.shape[1:] == mask.shape, (mask.shape, means.shape)

        with torch.no_grad():
            means, sigmas = torch.from_numpy(means), torch.from_numpy(sigmas)
            pi, mask = torch.from_numpy(pi), torch.from_numpy(mask)
            one_hot_sample = cls.categorical_pl_sample(pi, mask, random_seed)

            means = torch.sum(one_hot_sample * means, dim=0)[None, ...]
            sigmas = torch.sum(one_hot_sample * sigmas, dim=0)[None, ...]
            mask = mask[None, :, :]
            standard_sample = cls.normal_sample((n_samples, *means.shape[1:]), random_seed)
            adjusted_sample = (standard_sample * sigmas + means) * mask
        return adjusted_sample.numpy()

    @classmethod
    def tl_sample(cls, means: np.ndarray, sigmas: np.ndarray, pi: np.ndarray,
                  n_samples: int = 1000, random_seed: int = 42):
        """
        Tile level Sampling with Torch over large arrays.

        Args:
            means: [mixture_components, ] array
            sigmas: [mixture_components, ] array
            pi: [mixture_components, ] array
            random_seed: seed for random sampling
            n_samples: the number of sampling iterations
        Returns:
            the sampled values array with the shape [n_samples, 1]
        """
        assert means.shape == sigmas.shape == pi.shape, (means.shape, sigmas.shape, pi.shape)
        assert means.ndim == sigmas.ndim == pi.ndim == 1, (means.shape, sigmas.shape, pi.shape)

        with torch.no_grad():
            means, sigmas, pi = torch.from_numpy(means), torch.from_numpy(sigmas), torch.from_numpy(pi)
            one_hot_sample = cls.categorical_tl_sample(pi, random_seed)

            means = torch.sum(one_hot_sample * means, dim=0)
            sigmas = torch.sum(one_hot_sample * sigmas, dim=0)
            standard_sample = cls.normal_sample((n_samples,), random_seed)
            adjusted_sample = (standard_sample * sigmas + means)
        return adjusted_sample.numpy()
