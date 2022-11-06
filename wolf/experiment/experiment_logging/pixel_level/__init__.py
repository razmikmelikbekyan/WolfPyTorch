"""Package contains PIXEL LEVEL experiments loggers and visualizers for various task types."""
from .classification import *
from .regression import *
from ....enums import TaskTypes

PIXEL_LEVEL_LOGGERS = {
    TaskTypes.PL_REGRESSION: PLRegressionExperimentLogger,
    TaskTypes.PL_QUANTILE_REGRESSION: PLRegressionExperimentLogger,  # TODO we might want to log quantiles as well
    TaskTypes.PL_MDN_REGRESSION: PLRegressionExperimentLogger,  # TODO we might want to log pi and sigma as well
    TaskTypes.PL_BINARY_CLASSIFICATION: PLBinaryClassificationExperimentLogger,
}

PIXEL_LEVEL_VISUALIZERS = {
    TaskTypes.PL_REGRESSION: PLRegressionExperimentVisualizer,
    TaskTypes.PL_QUANTILE_REGRESSION: PLRegressionExperimentVisualizer,
    TaskTypes.PL_MDN_REGRESSION: PLRegressionExperimentVisualizer,
    TaskTypes.PL_BINARY_CLASSIFICATION: PLBinaryClassificationExperimentVisualizer,
}
