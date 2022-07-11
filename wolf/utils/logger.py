"""
Usage:
Import this module and use required levels like this:
import logger
logger.info("Started")
logger.error("Operation failed.")
logger.debug("Encountered debug case")
"""
import logging

from intelinair_utils import set_standard_logging_config

set_standard_logging_config()

logger = logging.getLogger('yield_forecasting')
