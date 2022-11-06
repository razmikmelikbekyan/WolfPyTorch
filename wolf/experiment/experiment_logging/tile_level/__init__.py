"""Package contains TILE LEVEL experiments loggers and visualizers for various task types."""
from .classification import *
from .regression import *
from ....enums import TaskTypes

TILE_LEVEL_LOGGERS = {
    TaskTypes.TL_REGRESSION: TLRegressionExperimentLogger,
    TaskTypes.TL_QUANTILE_REGRESSION: TLRegressionExperimentLogger,  # TODO we might want to log quantiles as well
    TaskTypes.TL_BINARY_CLASSIFICATION: TLClassificationExperimentLogger,
    TaskTypes.TL_MULTI_CLASSIFICATION: TLClassificationExperimentLogger
}

TILE_LEVEL_VISUALIZERS = {
    TaskTypes.TL_REGRESSION: TLRegressionExperimentVisualizer,
    TaskTypes.TL_QUANTILE_REGRESSION: TLRegressionExperimentVisualizer,
    TaskTypes.TL_BINARY_CLASSIFICATION: TLClassificationEpochVisualizer,
    TaskTypes.TL_MULTI_CLASSIFICATION: TLClassificationEpochVisualizer,
}
