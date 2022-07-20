import numpy as np
from typing import List, Dict, Tuple


def min_max_normalize(image: np.ndarray, skip_channels: List[int] = None, copy: bool = False,
                      by_channel: bool = True) -> np.ndarray:
    """
    Applies min - max normalization.
    Args:
        image: 2D or 3D image, numpy array, shape (H x W x C)
        skip_channels: the channels to skip normalization
        copy: if True, copies image before normalization
        by_channel: if True makes the normalization channel by channel,
                    otherwise will use the whole channels min and max

    Returns:
        normalized image
    """
    eps = 1e-6
    if image.ndim == 2:
        min_value, max_value = np.min(image), np.max(image)
        diff_value = max_value - min_value
        if diff_value < eps:
            out_image = image - min_value
        else:
            out_image = (image - min_value) / diff_value
    elif image.ndim == 3:
        if skip_channels:
            channels = tuple(sorted(set(range(image.shape[2])) - set(skip_channels)))
        else:
            channels = tuple(range(image.shape[2]))

        if by_channel:
            min_values, max_values = np.min(image, axis=(0, 1)), np.max(image, axis=(0, 1))
            diff_values = max_values - min_values

            min_channels, diff_channels = channels, channels
            diff_channels = tuple(i for i in diff_channels if diff_values[i] > eps)

            out_image = image.astype(np.float32, copy=copy)
            if min_channels:
                out_image[:, :, min_channels] = out_image[:, :, min_channels] - min_values[..., min_channels]
            if diff_channels:
                out_image[:, :, diff_channels] = out_image[:, :, diff_channels] / diff_values[..., diff_channels]
        else:
            min_value, max_value = np.min(image), np.max(image)
            diff_value = max_value - min_value

            min_channels, diff_channels = channels, channels
            diff_channels = tuple(i for i in diff_channels if diff_value > eps)

            out_image = image.astype(np.float32, copy=copy)
            if min_channels:
                out_image[:, :, min_channels] = out_image[:, :, min_channels] - min_value
            if diff_channels:
                out_image[:, :, diff_channels] = out_image[:, :, diff_channels] / diff_value
    else:
        raise ValueError(f'The input image must be 2D or 3D, got: {image.shape}')

    return out_image


def mean_std_normalize(image: np.ndarray, skip_channels: List[int] = None, copy: bool = False,
                       means_stds: Dict[int, Tuple[float, float]] or Tuple[float, float] = None) -> np.ndarray:
    """
    Applies mean - std normalization.
    Args:
        image: 2D or 3D image, numpy array, shape (H x W x C)
        skip_channels: the channels to skip normalization
        copy: if True, copies image before normalization
        means_stds: already given values for means and stds,
                    if 2D image: (mean, std)
                    if 3D image: {'channel': (mean, std)}
    Returns:
        normalized image
    """

    eps = 1e-6
    if image.ndim == 2:
        if means_stds:
            mean_value, std_value = means_stds
        else:
            mean_value, std_value = np.mean(image), np.std(image)
        if np.abs(std_value) < eps:
            out_image = image - mean_value
        else:
            out_image = (image - mean_value) / std_value
    elif image.ndim == 3:
        if means_stds and isinstance(means_stds, dict):
            means_stds = [tuple(means_stds.get(i, (0., 0.))) for i in range(image.shape[2])]
            mean_values, std_values = zip(*means_stds)
            mean_values, std_values = np.array(mean_values), np.array(std_values)
        elif means_stds and isinstance(means_stds, (tuple, list)):
            mean_value, std_value = means_stds
            mean_values = np.array([mean_value] * image.shape[2], dtype=np.float64)
            std_values = np.array([std_value] * image.shape[2], dtype=np.float64)
        else:
            mean_values = np.mean(image, axis=(0, 1), dtype=np.float64)
            std_values = np.std(image, axis=(0, 1), dtype=np.float64)

        if skip_channels:
            channels = tuple(sorted(set(range(image.shape[2])) - set(skip_channels)))
        else:
            channels = tuple(range(image.shape[2]))

        mean_channels, std_channels = channels, channels
        std_channels = tuple(i for i in std_channels if np.abs(std_values[i]) > eps)

        out_image = image.astype(np.float32, copy=copy)
        if mean_channels:
            out_image[:, :, mean_channels] = out_image[:, :, mean_channels] - mean_values[..., mean_channels]
        if std_channels:
            out_image[:, :, std_channels] = out_image[:, :, std_channels] / std_values[..., std_channels]
    else:
        raise ValueError(f'The input image must be 2D or 3D, got: {image.shape}')

    return out_image


def imagenet_normalize(image: np.ndarray, rgb_channels: Dict[str, int], copy: bool = False) -> np.ndarray:
    """
    Applies imagenet normalization: "mean_std" normalization by using
    predefined means and stds: [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225]

    Args:
        image: 2D or 3D image, numpy array, shape (H x W x C)
        rgb_channels: {'r': r_channel, 'g': g_channel, 'b': b_channel}
        copy: if True, copies image before normalization
    Returns:
        normalized image
    """
    imagenet_config = {'r': (0.485, 0.229), 'g': (0.456, 0.224), 'b': (0.406, 0.225)}
    rgb_channels = {k.lower(): v for k, v in rgb_channels.items()}
    if image.ndim == 3:
        means_stds = {rgb_channels[k]: v for k, v in imagenet_config.items()}
        return mean_std_normalize(image, copy=copy, means_stds=means_stds, skip_channels=None)
    else:
        raise ValueError(f'The input image must be 3D, got: {image.shape}')
