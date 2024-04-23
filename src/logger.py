"""
Logger's setup file
"""

import logging
import os.path

# base level of logging
LOGGING_LEVEL = logging.INFO

# name of the file where logs will be store in
# 'logs.log' by default
LOG_FILE_NAME = 'logs.log'

# full path to logs file
LOG_FILE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), LOG_FILE_NAME)


# logger BasicConfig that uses two handlers
# FileHandler - to write log-unit in file
# and StreamHadler to write in console
logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(levelname)s::%(asctime)s::%(module)s::%(funcName)s::%(filename)s::%(lineno)d %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='a')],
                    datefmt='%d-%b-%y %H:%M:%S'
                    )

# logger instance to import to another modules
logger = logging.getLogger(__name__)

