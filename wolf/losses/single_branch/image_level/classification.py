import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Loss

__all__ = [
    'BCELoss',
    'BCEWithLogitsLoss',
    'BinaryFocalLoss',
    'NLLLoss',
    'CrossEntropyLoss',
    'MultiClassFocalLoss',
    'LabelSmoothingLoss'
]


class BCELoss(nn.BCELoss, Loss):
    """Binary Cross Entropy.

    Should apply Sigmoid to prediction before using this loss.
    """
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, Loss):
    """Binary Cross Entropy with logits.

    It combines Sigmoid and BCE, so there is no need to apply Sigmoid to prediction.
    """
    pass


class NLLLoss(nn.NLLLoss, Loss):
    """Negative Log Likelihood.

    Should apply LogSoftmax to prediction before using this loss.
    """
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, Loss):
    """Cross Entropy Loss.

    It combines LogSoftmax and NLLLoss, so there is no need to apply LogSoftmax to prediction.
    """
    pass


class MultiClassFocalLoss(nn.CrossEntropyLoss, Loss):
    """
      This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
      'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
         Focal_Loss= -1*alpha_{t}*((1-p_{t})^gamma)*log(p_{t})

      This loss expect row prediction - logits.
      """

    def __init__(self, weight: torch.Tensor = None, gamma: int = 2):
        nn.CrossEntropyLoss.__init__(self, weight=weight)
        Loss.__init__(self)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Args:
            y_pred: net output with shape [B x NClasses]
            y_true: ground truth with shape [B]

        Returns: the scalar loss value averaged over the batch
        """
        logpt = F.cross_entropy(y_pred, y_true, reduction='none', weight=self.weight)
        return (((1 - torch.exp(-logpt)) ** self.gamma) * logpt).mean()


class LabelSmoothingLoss(Loss):
    """Labels Smoothing loss: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch

    This loss expect row prediction - logits.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean", weight: torch.Tensor = None):
        super(LabelSmoothingLoss, self).__init__(name='LabelSmoothingLoss')
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss: torch.Tensor):
        return loss.mean() if self.reduction == 'mean' else (loss.sum() if self.reduction == 'sum' else loss)

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(y_pred.device)

        n = y_pred.size(-1)
        log_preds = F.log_softmax(y_pred, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(log_preds, y_true, reduction=self.reduction, weight=self.weight)
        return self.linear_combination(loss / n, nll)


class BinaryFocalLoss(nn.BCELoss, Loss):
    """This is a implementation of Focal Loss for binary classification problem."""

    def __init__(self, weight: torch.Tensor = None, gamma: int = 2, with_logit: bool = True, reduction: str = 'mean'):
        """

        Args:
            weight: weight parameter will act as the alpha parameter to balance class weights
            gamma: Power factor for dampening weight (focal strength)
            with_logit: False if model's output is a result of sigmoid function ele True
            reduction: there are two options: If 'sum', it is calculated sum of loss, otherwise mean of loss
        """
        nn.BCELoss.__init__(self, weight=weight)
        Loss.__init__(self)
        self.gamma = gamma
        self.with_logit = with_logit
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Args:
            y_pred: net output with shape [B]
            y_true: ground truth with shape [B]

        Returns: the loss value
        """
        y_true = y_true.type_as(y_pred)

        if not self.with_logit:
            logpt = F.binary_cross_entropy(y_pred, y_true, reduction="none", weight=self.weight)
        else:
            logpt = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none", weight=self.weight)

        pt = torch.exp(-logpt)

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * logpt

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum(dtype=torch.float32)

        return loss
