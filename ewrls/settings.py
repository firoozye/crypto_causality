import datetime
import logging.handlers

import pandas as pd
import os
from pathlib import Path
"""

LOGGER Setup

"""

LOGGER = logging.getLogger(__name__)

# TODO: Threading and multi-processing. Which run wrote that entry?
# TODO: consider using logging.conf file!
# .basicConfig is neater than current setup
ERROR_DIRECTORY = './logs/'
# TODO: Make log file save to output_dir

date = datetime.date.today().strftime('%Y%m%d')

LOG_FILENAME = ERROR_DIRECTORY + f'run_{date}.log'
# Add the log message handler to the logger
formatter = logging.Formatter(
    fmt=
    f"[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S")

handler = logging.handlers.RotatingFileHandler(LOG_FILENAME, backupCount=5)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.info('STARTING UP LOGGER')
'''
Set  up multiple logger files for each 'submodule'?
https://stackoverflow.com/questions/17035077/
logging-to-multiple-log-files-from-different-classes-in-python
'''
'''
Bond Universe settings
'''

bond_data_date_format = 'yyyy-mm-dd'

# NOTE: No need to change this if we have a 20y
# all_bond_terms =  {2, 3, 5, 7, 10, 20, 30}
# can set up overwritten files in handler or backup ones
