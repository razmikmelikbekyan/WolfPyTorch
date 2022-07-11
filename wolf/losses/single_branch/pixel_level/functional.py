from typing import Iterable

import torch


def take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x._device)) for x in xs]
        return xs


def threshold_prediction(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def get_iou(y_pred: torch.Tensor,
            y_true: torch.Tensor,
            eps: float = 1e-7,
            threshold: int or float = None,
            ignore_channels: Iterable = None) -> float:
    """
    Calculates Intersection over Union score (Jaccard score) between ground truth and prediction.
    Args:
        y_pred: predicted tensor
        y_true: ground truth tensor
        eps: epsilon to avoid zero division
        threshold: threshold used for outputs binarization
        ignore_channels: the iterable with channels to ignore from calculation
    Returns:
        float: IoU (Jaccard) score
    """
    assert y_pred.shape == y_true.shape

    y_pred = threshold_prediction(y_pred, threshold=threshold)
    y_pred, y_true = take_channels(y_pred, y_true, ignore_channels=ignore_channels)

    intersection = torch.einsum('bcwh, bcwh->b', y_true, y_pred)
    union = torch.einsum('bcwh->b', y_true) + torch.einsum('bcwh->b', y_pred) - intersection
    score = (intersection + eps) / (union + eps)
    return score.mean()


def get_f_score(y_pred: torch.Tensor,
                y_true: torch.Tensor,
                beta: float = 1,
                eps: float = 1e-7,
                threshold: float = None,
                ignore_channels: Iterable = None) -> float:
    """Calculates F-score between ground truth and prediction.
    Args:
        y_pred: predicted tensor, [BatchSize x 1 x W x H]
        y_true: ground truth tensor, [BatchSize x 1 x W x H]
        beta: positive constant for calculation of f score
        eps: epsilon to avoid zero division
        threshold: threshold used for outputs binarization
        ignore_channels: the iterable with channels to ignore from calculation
    Returns:
        float: F score
    """
    assert y_pred.shape == y_true.shape

    y_pred = threshold_prediction(y_pred, threshold=threshold)
    y_pred, y_true = take_channels(y_pred, y_true, ignore_channels=ignore_channels)

    tp = torch.einsum('bcwh, bcwh->b', y_true, y_pred)  # intersection area
    fp = torch.einsum('bcwh->b', y_pred) - tp  # total predicted positive area - intersection area
    fn = torch.einsum('bcwh->b', y_true) - tp  # total actual positive area - intersection area

    score = (
            ((1 + beta ** 2) * tp + eps) /
            ((1 + beta ** 2) * tp + (beta ** 2) * fn + fp + eps)
    )
    return score.mean()


def get_accuracy(y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 threshold: float = 0.5,
                 ignore_channels: Iterable = None):
    """
    Calculates pixel-wise Accuracy score between ground truth and prediction.
    Args:
        y_pred: predicted tensor, [BatchSize x 1 x W x H]
        y_true: ground truth tensor, [BatchSize x 1 x W x H]
        threshold: threshold used for outputs binarization
        ignore_channels: the iterable with channels to ignore from calculation
    Returns:
        float: precision score
    """
    assert y_pred.shape == y_true.shape

    y_pred = threshold_prediction(y_pred, threshold=threshold)
    y_pred, y_true = take_channels(y_pred, y_true, ignore_channels=ignore_channels)

    _, _, w, h = y_true.shape

    correct_prediction = torch.einsum('bcwh->b', y_true == y_pred)
    score = correct_prediction.float() / float(w * h)
    return score.mean()


def get_precision(y_pred: torch.Tensor,
                  y_true: torch.Tensor,
                  eps: float = 1e-7,
                  threshold: float = 0.5,
                  ignore_channels: Iterable = None):
    """
    Calculates Precision score between ground truth and prediction.
    Args:
        y_pred: predicted tensor, [BatchSize x 1 x W x H]
        y_true: ground truth tensor, [BatchSize x 1 x W x H]
        eps: epsilon to avoid zero division
        threshold: threshold used for outputs binarization
        ignore_channels: the iterable with channels to ignore from calculation
    Returns:
        float: precision score
    """
    assert y_pred.shape == y_true.shape

    y_pred = threshold_prediction(y_pred, threshold=threshold)
    y_pred, y_true = take_channels(y_pred, y_true, ignore_channels=ignore_channels)

    tp = torch.einsum('bcwh, bcwh->b', y_true, y_pred)
    fp = torch.einsum('bcwh->b', y_pred) - tp

    score = (tp + eps) / (tp + fp + eps)
    return score.mean()


def get_recall(y_pred: torch.Tensor,
               y_true: torch.Tensor,
               eps: float = 1e-7,
               threshold: float = 0.5,
               ignore_channels: Iterable = None):
    """
    Calculates Recall score between ground truth and prediction.
    Args:
        y_pred: predicted tensor, [BatchSize x 1 x W x H]
        y_true: ground truth tensor, [BatchSize x 1 x W x H]
        eps: epsilon to avoid zero division
        threshold: threshold used for outputs binarization
        ignore_channels: the iterable with channels to ignore from calculation
    Returns:
        float: precision score
    """
    assert y_pred.shape == y_true.shape

    y_pred = threshold_prediction(y_pred, threshold=threshold)
    y_pred, y_true = take_channels(y_pred, y_true, ignore_channels=ignore_channels)

    tp = torch.einsum('bcwh, bcwh->b', y_true, y_pred)
    fn = torch.einsum('bcwh->b', y_true) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score.mean()
