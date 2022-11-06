from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from torchmetrics.functional import auroc, accuracy, confusion_matrix

from ...base import BaseEvaluator


class BaseClassificationEvaluator(BaseEvaluator):
    """A Base class for all classification evaluators."""

    ConfusionMatrixMetrics: frozenset
    ScoreRequiringMetrics: frozenset

    MetricsPool: frozenset

    AverageTypes = ('by_class', 'binary', 'micro', 'macro', 'weighted')
    PredictionTypes = frozenset(['logit', 'probability', 'label'])

    def __init__(self,
                 metrics: List[Tuple[str, Dict]],
                 n_classes: int,
                 prediction_type: str,
                 classes: List,
                 device: torch.device = torch.device("cpu"),
                 class_names: List[str] = None,
                 threshold: float = 0.5):
        """

        Args:
            metrics: the names of metrics and respective kwargs for calculating them
            n_classes: the number of classes
            prediction_type: the type of prediction: 'logit', 'probability', 'label'
            classes: the list of class labels: [1, 0] (the confusion matrix rows will be created based on this order)
            class_names: the list of strings which represents the class names,
                         it must be in the same order as given classes
            threshold: the threshold to use in case of binary classification
        """
        super(BaseClassificationEvaluator, self).__init__(metrics=metrics)
        if prediction_type not in self.PredictionTypes:
            raise ValueError(
                f"Given prediction type={prediction_type} is wrong, please select from {list(self.PredictionTypes)}"
            )
        assert len(classes) >= 2 or len(classes) != n_classes, classes

        self._n_classes = n_classes
        self._prediction_type = prediction_type
        self._classes = classes
        self._class_names = self._get_class_names(classes, class_names)
        self._threshold = threshold
        self._validate()

        self._history_is_required = any(x in self.ScoreRequiringMetrics for x, _ in metrics)

        self.register_buffer('cfm', torch.zeros((n_classes, n_classes), dtype=torch.float32))
        self.register_buffer('n_samples', torch.tensor(0., dtype=torch.float32))

        if prediction_type == 'label' and self._history_is_required:
            print('prediction_type="label" and score metrics are mutually exclusive, disabling score metrics.')
            self._history_is_required = False

        if self._history_is_required:
            # adding history for history requiring metrics
            self.y_pred = list()
            self.y_true = list()

        self._state = {}

        self.to(device=device)

    def _validate(self):
        for metric, metric_kwargs in self._metrics:
            try:
                average_type = metric_kwargs['average_type']
            except KeyError:
                continue
            else:
                if average_type == 'binary':
                    if len(self._classes) != 2 or self._classes[0] != 1:
                        raise ValueError(f"Binary average type assumes that classes must be 2 and the first "
                                         f"one must be the positive class, got: {self._classes}")

    @staticmethod
    def _get_class_names(classes: List, class_names: Dict or List[str]) -> List[str]:
        """Performs some sanity checks and returns class_names."""

        if not class_names:
            class_names = [f'class_{i}' for i in range(1, len(classes) + 1)]

        if len(class_names) != len(classes):
            raise ValueError("Provided classes and class_names have different sizes.")

        if not isinstance(class_names, list):
            raise ValueError(f'class_names must be list, got {type(class_names)}')

        return class_names

    @property
    def classes(self) -> List:
        return self._classes

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @staticmethod
    def get_confusion_matrix(cfm: torch.Tensor) -> torch.Tensor:
        return cfm

    @staticmethod
    def get_accuracy(cfm: torch.Tensor) -> torch.Tensor:
        return torch.trace(cfm) / torch.sum(cfm)

    @staticmethod
    def get_balanced_accuracy(cfm: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.diagonal(cfm) / torch.sum(cfm, dim=1))

    @classmethod
    def get_precision_recall(cls, cfm: torch.Tensor, average_type: str, axis: int) -> torch.Tensor:

        if average_type == 'by_class':
            return torch.diagonal(cfm) / torch.maximum(torch.sum(cfm, dim=axis), torch.tensor(1e-6).to(cfm.device))
        elif average_type == 'binary':
            return cls.get_precision_recall(cfm, 'by_class', axis)[0]
        elif average_type == 'macro':
            return torch.mean(cls.get_precision_recall(cfm, 'by_class', axis))  # the mean of each class
        elif average_type == 'micro':
            return torch.sum(torch.diagonal(cfm)) / torch.sum(cfm)  # calculating as a total score
        elif average_type == 'weighted':
            weights = torch.sum(cfm, dim=1) / torch.sum(cfm)
            by_class = cls.get_precision_recall(cfm, 'by_class', axis)
            return torch.sum(weights * by_class)
        else:
            raise ValueError(f'Given {average_type} average method is wrong.')

    @classmethod
    def get_recall(cls, cfm: torch.Tensor, average_type: str) -> torch.Tensor:
        return cls.get_precision_recall(cfm=cfm, average_type=average_type, axis=1)

    @classmethod
    def get_precision(cls, cfm: torch.Tensor, average_type: str) -> torch.Tensor:
        return cls.get_precision_recall(cfm=cfm, average_type=average_type, axis=0)

    @classmethod
    def get_f_score(cls, cfm: torch.Tensor, average_type: str, betta: float) -> torch.Tensor:
        if average_type in ('by_class', 'micro'):
            eps = 1e-6
            precision = cls.get_precision(cfm, average_type)
            recall = cls.get_recall(cfm, average_type)
            nominator = (1. + (betta ** 2)) * (precision * recall)
            denominator = ((betta ** 2) * precision + recall)
            if isinstance(denominator, float):
                denominator = 1. if denominator <= eps else denominator
            else:
                denominator[denominator <= eps] = 1.
            return nominator / denominator
        if average_type == 'binary':
            return cls.get_f_score(cfm, 'by_class', betta)[0]
        if average_type == 'macro':
            return torch.mean(cls.get_f_score(cfm, 'by_class', betta))
        elif average_type == 'weighted':
            weights = torch.sum(cfm, dim=1) / torch.sum(cfm)
            return torch.sum(weights * cls.get_f_score(cfm, 'by_class', betta))
        else:
            raise ValueError(f'Given {average_type} average method is wrong.')

    @classmethod
    def get_iou(cls, cfm: torch.Tensor, average_type: str) -> torch.Tensor:
        if average_type == 'by_class':
            intersection = torch.diagonal(cfm)
            union = torch.sum(cfm, dim=1) + torch.sum(cfm, dim=0) - intersection
            return intersection / union
        elif average_type == 'binary':
            return cls.get_iou(cfm, average_type='by_class')[0]
        elif average_type == 'macro':
            return torch.mean(cls.get_iou(cfm, average_type='by_class'))
        elif average_type == 'micro':
            total_intersection = torch.sum(torch.diagonal(cfm))
            total_union = torch.sum(torch.sum(cfm, dim=1) + torch.sum(cfm, dim=0) - torch.diagonal(cfm))
            return total_intersection / total_union
        elif average_type == 'weighted':
            weights = torch.sum(cfm, dim=1) / torch.sum(cfm)
            by_class = cls.get_iou(cfm, average_type='by_class')
            return torch.sum(weights * by_class)
        else:
            raise ValueError(f'Given {average_type} average method is wrong.')

    @classmethod
    def get_f1(cls, cfm: torch.Tensor, average_type: str) -> torch.Tensor:
        return cls.get_f_score(cfm=cfm, average_type=average_type, betta=1.)

    @staticmethod
    def get_roc_auc_score(y_true: torch.Tensor, y_pred: torch.Tensor, average_type: str, **kwargs) -> torch.Tensor:
        return auroc(target=y_true, preds=y_pred, average=average_type, **kwargs)

    @staticmethod
    def get_top_k_accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int, **kwargs) -> torch.Tensor:
        return accuracy(target=y_true, preds=y_pred, top_k=top_k, **kwargs)

    def _add_class_names(self, metric_value: float or torch.Tensor, metric_kwargs: Dict) -> float or Dict or np.ndarray:
        """Adds class names to metric."""
        by_class = metric_kwargs.get('average_type', None) == 'by_class'

        if by_class and isinstance(metric_value, torch.Tensor):
            assert len(metric_value) == len(self._classes)
            return {
                (self._class_names[i] if self._class_names else x): y
                for i, (x, y) in enumerate(zip(self._classes, metric_value))
            }
        else:
            return metric_value

    def _transform_prediction(self, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Transforms prediction to the respective class label for further calculation of metrics."""

        if self._prediction_type == 'logit':
            if self._n_classes == 2:
                y_pred = torch.sigmoid(y_pred)
                y_pred_labels = (y_pred > self._threshold).to(torch.int64)
            else:
                y_pred = F.softmax(y_pred, dim=1)
                y_pred_labels = torch.argmax(y_pred, dim=1)

        elif self._prediction_type == 'probability':
            if self._n_classes == 2:
                y_pred_labels = (y_pred > self._threshold).to(torch.int64)
            else:
                y_pred_labels = torch.argmax(y_pred, dim=1)

        else:
            y_pred_labels = y_pred.to(torch.int64)
        return y_pred, y_pred_labels

    def compute(self) -> Dict:
        for metric_name, metric_kwargs in self._metrics:

            if metric_kwargs:
                saving_name = f'{metric_name}-{"-".join(f"{k}={v}" for k, v in metric_kwargs.items())}'
            else:
                saving_name = metric_name

            if metric_name in self.ConfusionMatrixMetrics:
                args = (self.cfm,)
            else:
                args = (torch.cat(self.y_true, dim=0), torch.cat(self.y_pred, dim=0))

            f = getattr(self, f'get_{metric_name}')
            try:
                metric_value = f(*args, **metric_kwargs)
            except TypeError:
                metric_value = f(*args)

            self._state[saving_name] = self._add_class_names(metric_value, metric_kwargs)
            self._state['class_names'] = self.class_names

        return self.state

    def reset(self):
        self.cfm.zero_()
        self.n_samples.zero_()
        self.y_pred = []
        self.y_true = []

    @property
    def state(self) -> Dict:
        return self.to_python_float(deepcopy(self._state))

    @property
    def scores(self) -> Dict:
        """
        A JSON-serializable version of the state dict to be directly provided
        as the 'scores' argument to intelinair_ml.evaluation.create_dataset_evaluation
        """

        return dict(map(lambda k, v: (k, v.tolist() if hasattr(v, 'tolist') else v), *zip(*self.state.items())))

    def _check_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Checks the input shapes."""
        pass

    def update(self, y_pred: torch.Tensor, y_true: torch.IntTensor):
        """Updates the state of the buffer by accumulating given predictions and labels."""
        self._check_inputs(y_pred, y_true)
        y_pred, y_pred_labels = self._transform_prediction(y_pred)

        cfm = confusion_matrix(target=y_true, preds=y_pred_labels, num_classes=len(self.classes))
        cfm = cfm[self.classes, :][:, self.classes]  # ordering cfm according to our classes

        self.n_samples += len(y_true)
        self.cfm += cfm

        if self._history_is_required:
            self.y_pred.append(y_pred)
            self.y_true.append(y_true)


class TLClassificationEvaluator(BaseClassificationEvaluator):
    """Special class for evaluation TL Classification models."""

    ConfusionMatrixMetrics = frozenset([
        'accuracy',
        'balanced_accuracy',
        'confusion_matrix',
        'recall',
        'precision',
        'f1',
        'iou'
    ])

    ScoreRequiringMetrics = frozenset([
        'roc_auc_score',
        'top_k_accuracy_score'
    ])

    MetricsPool = ConfusionMatrixMetrics | ScoreRequiringMetrics

    def _check_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Checks the input shapes.

        Predictions and targets are supposed to fall into one of these categories (and are
        validated to make sure this is the case):

        * Both y_pred and y_true are of shape ``(B,)``, and both are integers (multi-class)
        * Both y_pred and y_true are of shape ``(B,)``, and y_true is binary, while y_pred
          are a float (binary)
        * y_pred are of shape ``(B, C)`` and are floats, and y_true is of shape ``(B,)`` and
          is integer (multi-class)
        """

        if y_true.dim() != 1:
            raise ValueError(f"True labels must have (batch size, ) shape, got: {y_true.shape}")

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"True labels and predictions must have the same batch sizes, "
                             f"got: y_true={y_true.shape}, y_pred={y_pred.shape}")

        if self._prediction_type == 'label':
            if y_pred.dim() != 1:
                raise ValueError("In case the prediction_type=='label' the prediction should one dimensional "
                                 f"integers with shape (B, ), got: {y_pred.shape}")
        else:
            if self._n_classes == 2:  # binary classification
                if y_pred.dim() != 1 or (y_pred.dim() == 2 and y_pred.shape[1] != 1):
                    raise ValueError("In case of binary classification the prediction shape must be (B, 1) or (B, ), "
                                     f"got: {y_pred.shape}.")
            else:
                if y_pred.dim() != 2 and y_pred.shape[1] != self._n_classes:
                    raise ValueError("In case of multiclass classification the prediction shape must be (B, C), "
                                     f"got: {y_pred.shape}.")
