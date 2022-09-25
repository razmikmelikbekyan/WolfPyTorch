"""Package contains PIXEL LEVEL evaluators for various task types."""
from .classifcation import PLClassificationEvaluator
from .regression import PLRegressionEvaluator

from ....constants import TaskTypes

PIXEL_LEVEL_EVALUATORS = {
    TaskTypes.PL_REGRESSION: PLRegressionEvaluator,
    TaskTypes.PL_QUANTILE_REGRESSION: PLRegressionEvaluator,
    TaskTypes.PL_MDN_REGRESSION: PLRegressionEvaluator,
    TaskTypes.PL_BINARY_CLASSIFICATION: PLClassificationEvaluator,
    TaskTypes.PL_MULTI_CLASSIFICATION: PLClassificationEvaluator
}
