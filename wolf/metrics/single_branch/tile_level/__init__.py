"""Package contains TILE LEVEL evaluators for various task types."""
from .classification import TLClassificationEvaluator
from .regression import TLRegressionEvaluator
from ....constants import TaskTypes

TILE_LEVEL_EVALUATORS = {
    TaskTypes.TL_BINARY_CLASSIFICATION: TLClassificationEvaluator,
    TaskTypes.TL_MULTI_CLASSIFICATION: TLClassificationEvaluator,
    TaskTypes.TL_REGRESSION: TLRegressionEvaluator,
    TaskTypes.TL_QUANTILE_REGRESSION: TLRegressionEvaluator,
    TaskTypes.TL_MDN_REGRESSION: TLRegressionEvaluator
}
