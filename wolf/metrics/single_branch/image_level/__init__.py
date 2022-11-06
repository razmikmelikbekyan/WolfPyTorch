"""Package contains IMAGE LEVEL evaluators for various task types."""
from .classification import ILClassificationEvaluator
from .regression import ILRegressionEvaluator
from ....enums import TaskTypes

IMAGE_LEVEL_EVALUATORS = {
    TaskTypes.IL_BINARY_CLASSIFICATION: ILClassificationEvaluator,
    TaskTypes.IL_MULTI_CLASSIFICATION: ILClassificationEvaluator,
    TaskTypes.IL_REGRESSION: ILRegressionEvaluator,
    TaskTypes.IL_QUANTILE_REGRESSION: ILRegressionEvaluator,
    TaskTypes.IL_MDN_REGRESSION: ILRegressionEvaluator
}
