from typing import Dict

import torch.nn as nn
from segmentation_models_pytorch.losses import (
    FocalLoss, DiceLoss, JaccardLoss, TverskyLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss,
    BINARY_MODE, MULTICLASS_MODE,
)

from ..base import Loss, SumOfLosses

__all__ = [
    'BCELoss',
    'BCEWithLogitsLoss',
    'BinaryJaccardLoss',
    'BinaryDiceLoss',
    'BinaryFocalLoss',
    'BinaryTverskyLoss',
    'BCEDiceLoss',
    'BCEDiceLogitsLoss',
    'CrossEntropyLoss',
    'MultiClassJaccardLoss',
    'MultiClassDiceLoss',
    'MultiClassFocalLoss',
    'MultiClassTverskyLoss',
]


# ---------------------------- Binary Losses ----------------------------

class BCELoss(nn.BCELoss, Loss):
    """Binary Cross Entropy.

    Should apply Sigmoid to prediction before using this loss.
    """
    pass


class BCEWithLogitsLoss(SoftBCEWithLogitsLoss, Loss):
    """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing"""
    pass


class BinaryJaccardLoss(JaccardLoss, Loss):
    """Jaccard (IoU) loss implementation for 2 classes case.

    Please pay attention to `from_logits` parameter: in case of True it expects the un-normalized probabilities.
    """

    def __init__(self, *args, **kwargs):
        JaccardLoss.__init__(self, BINARY_MODE, *args, **kwargs)
        Loss.__init__(self, name="BinaryJaccardLoss")


class BinaryDiceLoss(DiceLoss, Loss):
    """
    Dice loss (f1) implementation for 2 classes case.

    Please pay attention to `from_logits` parameter: in case of True it expects the un-normalized probabilities.
    """

    def __init__(self, *args, **kwargs):
        DiceLoss.__init__(self, BINARY_MODE, *args, **kwargs)
        Loss.__init__(self, name="BinaryDiceLoss")


class BinaryTverskyLoss(TverskyLoss, Loss):
    """
    Tversky loss implementation for 2 classes case.

    Please pay attention to `from_logits` parameter: in case of True it expects the un-normalized probabilities.
    """

    def __init__(self, *args, **kwargs):
        TverskyLoss.__init__(self, BINARY_MODE, *args, **kwargs)
        Loss.__init__(self, name="BinaryTverskyLoss")


class BinaryFocalLoss(FocalLoss, Loss):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
       Focal_Loss= -1*alpha_{t}*((1-p_{t})^gamma)*log(p_{t})

    Expects the target before Sigmoid - aka raw logits.
    """

    def __init__(self, *args, **kwargs):
        FocalLoss.__init__(self, BINARY_MODE, *args, **kwargs)
        Loss.__init__(self, name="BinaryFocalLoss")


class BCEDiceLoss(SumOfLosses):
    """
    Binary Cross Entropy + Dice Loss

    Expects the target after Sigmoid.
    """

    def __init__(self, bce_kwargs: Dict = None, dice_kwargs: Dict = None):
        bce_kwargs = {} if bce_kwargs is None else bce_kwargs
        dice_kwargs = {} if dice_kwargs is None else dice_kwargs
        super().__init__(l1=BCELoss(**bce_kwargs), l2=BinaryDiceLoss(**dice_kwargs))


class BCEDiceLogitsLoss(SumOfLosses):
    """
    Binary Cross Entropy + Dice Loss

    Expects the target before Sigmoid - aka raw logits.
    """

    def __init__(self, bce_kwargs: Dict = None, dice_kwargs: Dict = None):
        bce_kwargs = {} if bce_kwargs is None else bce_kwargs
        dice_kwargs = {} if dice_kwargs is None else dice_kwargs
        super().__init__(l1=BCEWithLogitsLoss(**bce_kwargs), l2=BinaryDiceLoss(**dice_kwargs))


# ---------------------------- Multi Class Losses ----------------------------


class CrossEntropyLoss(SoftCrossEntropyLoss, Loss):
    """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing"""
    pass


class MultiClassJaccardLoss(JaccardLoss, Loss):
    """Jaccard (IoU) loss implementation for multiple classes case.

    Please pay attention to `from_logits` parameter: in case of True it expects the un-normalized probabilities.
    """

    def __init__(self, *args, **kwargs):
        JaccardLoss.__init__(self, MULTICLASS_MODE, *args, **kwargs)
        Loss.__init__(self, name="MultiClassJaccardLoss")


class MultiClassDiceLoss(DiceLoss, Loss):
    """
    Dice loss (f1) implementation for multiple classes case.

    Please pay attention to `from_logits` parameter: in case of True it expects the un-normalized probabilities.
    """

    def __init__(self, *args, **kwargs):
        DiceLoss.__init__(self, MULTICLASS_MODE, *args, **kwargs)
        Loss.__init__(self, name="MultiClassDiceLoss")


class MultiClassFocalLoss(FocalLoss, Loss):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
       Focal_Loss= -1*alpha_{t}*((1-p_{t})^gamma)*log(p_{t})

    Expects the target before Softmax - aka raw logits.
    """

    def __init__(self, *args, **kwargs):
        FocalLoss.__init__(self, MULTICLASS_MODE, *args, **kwargs)
        Loss.__init__(self, name="MultiClassFocalLoss")


class MultiClassTverskyLoss(DiceLoss, Loss):
    """
    Tversky loss implementation for multiple classes case.

    Please pay attention to `from_logits` parameter: in case of True it expects the un-normalized probabilities.
    """

    def __init__(self, *args, **kwargs):
        TverskyLoss.__init__(self, MULTICLASS_MODE, *args, **kwargs)
        Loss.__init__(self, name="MultiClassTverskyLoss")
