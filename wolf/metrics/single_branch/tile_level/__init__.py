"""Package contains TILE LEVEL evaluators for various task types."""
from ....constants import TaskTypes
from intelinair_ml.evaluators.regression import RegressionEvaluator
from intelinair_ml.evaluators.classification import ClassificationEvaluator

TILE_LEVEL_EVALUATORS = {
    TaskTypes.TL_BINARY_CLASSIFICATION: ClassificationEvaluator,
    TaskTypes.TL_MULTI_CLASSIFICATION: ClassificationEvaluator,
    TaskTypes.TL_REGRESSION: RegressionEvaluator,
    TaskTypes.TL_QUANTILE_REGRESSION: RegressionEvaluator,
    TaskTypes.TL_MDN_REGRESSION: RegressionEvaluator
}
