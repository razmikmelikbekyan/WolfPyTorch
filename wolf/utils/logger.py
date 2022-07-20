"""
Usage:
Import this module and use required levels like this:
import logger
logger.info("Started")
logger.error("Operation failed.")
logger.debug("Encountered debug case")
"""
import logging

logger = logging.getLogger('wolf')
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)

# format handlers
formatter = logging.Formatter(
    fmt='%(filename)s | %(lineno)d | %(funcName)s | %(asctime)s | %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
consoleHandler.setFormatter(formatter)

# adding handlers to the logger
logger.addHandler(consoleHandler)
