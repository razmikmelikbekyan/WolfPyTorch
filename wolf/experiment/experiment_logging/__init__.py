from typing import Type

from .base_logger import BaseExperimentLogger
from .base_visualizer import BaseExperimentsResultVisualizer
from .pixel_level import PIXEL_LEVEL_LOGGERS, PIXEL_LEVEL_VISUALIZERS
from .image_level import IMAGE_LEVEL_LOGGERS, IMAGE_LEVEL_VISUALIZERS
from ...enums import TaskTypes


def get_experiment_logger(task: TaskTypes) -> Type[BaseExperimentLogger]:
    """Returns the experiment Logger for given task."""
    if task in TaskTypes.get_image_level_tasks():
        loggers_pool = IMAGE_LEVEL_LOGGERS
    elif task in TaskTypes.get_pixel_level_tasks():
        loggers_pool = PIXEL_LEVEL_LOGGERS
    else:
        raise NotImplementedError(f"For task={task} there are no implemented Experiment loggers.")

    try:
        experiment_logger = loggers_pool[task]
    except KeyError:
        raise ValueError(
            f'Given task={task} is not supported.'
            f'Please select from: {" | ".join(loggers_pool.keys())}.'
        )
    return experiment_logger


def get_experiment_visualizer(task: TaskTypes) -> Type[BaseExperimentsResultVisualizer]:
    """Returns the experiment Visualizer for given task."""
    if task in TaskTypes.get_image_level_tasks():
        visualizers_pool = IMAGE_LEVEL_VISUALIZERS
    elif task in TaskTypes.get_pixel_level_tasks():
        visualizers_pool = PIXEL_LEVEL_VISUALIZERS
    else:
        raise NotImplementedError(f"For task={task} there are no implemented Experiment visualizers..")

    try:
        visualizer = visualizers_pool[task]
    except KeyError:
        raise ValueError(
            f'Given task={task} is not supported.'
            f'Please select from: {" | ".join(visualizers_pool.keys())}.'
        )
    return visualizer
