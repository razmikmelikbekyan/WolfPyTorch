"""Package contains TILE LEVEL evaluators for various task types."""
from .classification import TLClassificationEvaluator
from .regression import TLRegressionEvaluator
from ....enums import TaskTypes

TILE_LEVEL_EVALUATORS = {
    TaskTypes.IL_BINARY_CLASSIFICATION: TLClassificationEvaluator,
    TaskTypes.IL_MULTI_CLASSIFICATION: TLClassificationEvaluator,
    TaskTypes.IL_REGRESSION: TLRegressionEvaluator,
    TaskTypes.IL_QUANTILE_REGRESSION: TLRegressionEvaluator,
    TaskTypes.IL_MDN_REGRESSION: TLRegressionEvaluator
}
