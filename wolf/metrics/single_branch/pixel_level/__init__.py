"""Package contains PIXEL LEVEL evaluators for various task types."""
from intelinair_ml.evaluators.classification import SegmentationEvaluator
from intelinair_ml.evaluators.regression import RegressionSegmentationEvaluator

from ....constants import TaskTypes

PIXEL_LEVEL_EVALUATORS = {
    TaskTypes.PL_REGRESSION: RegressionSegmentationEvaluator,
    TaskTypes.PL_QUANTILE_REGRESSION: RegressionSegmentationEvaluator,
    TaskTypes.PL_MDN_REGRESSION: RegressionSegmentationEvaluator,
    TaskTypes.PL_BINARY_CLASSIFICATION: SegmentationEvaluator,
    TaskTypes.PL_MULTI_CLASSIFICATION: SegmentationEvaluator
}
