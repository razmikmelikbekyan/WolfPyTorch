"""This package represents the main package for training and evaluation of deep learning models."""

from .enums import *
from .dataset import *
from .epoch import *
from .experiment import *
from .losses import Loss, get_loss, MultiBranchLoss, TemporalLoss
from .metrics import get_evaluator, BaseEvaluator, MultiBranchEvaluator
from .optimization import get_lr_scheduler, get_warmup_scheduler, get_optimizer
from .regularization import get_regularizer, WeightsRegularization
