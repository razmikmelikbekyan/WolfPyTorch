"""Package contains IMAGE LEVEL experiments loggers and visualizers for various task types."""
from .classification import *
from .regression import *
from ....enums import TaskTypes

IMAGE_LEVEL_LOGGERS = {
    TaskTypes.IL_REGRESSION: ILRegressionExperimentLogger,
    TaskTypes.IL_QUANTILE_REGRESSION: ILRegressionExperimentLogger,  # TODO we might want to log quantiles as well
    TaskTypes.IL_BINARY_CLASSIFICATION: ILClassificationExperimentLogger,
    TaskTypes.IL_MULTI_CLASSIFICATION: ILClassificationExperimentLogger
}

IMAGE_LEVEL_VISUALIZERS = {
    TaskTypes.IL_REGRESSION: ILRegressionExperimentVisualizer,
    TaskTypes.IL_QUANTILE_REGRESSION: ILRegressionExperimentVisualizer,
    TaskTypes.IL_BINARY_CLASSIFICATION: ILClassificationEpochVisualizer,
    TaskTypes.IL_MULTI_CLASSIFICATION: ILClassificationEpochVisualizer,
}
