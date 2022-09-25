"""The package represents the base experiment that does not depend on YieldForecasting."""
from .config import BaseExperimentConfig
from .parser import BaseExperimentConfigParser
from .experiment import BaseExperiment, run_experiment
from .experiment_logging import *

# TODO: get rid of opencv from requirements and replace with PIL
# TODO: implement read_image method in base dataset
# TODO: replace `input_image` and others with ENUM
