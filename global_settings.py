""" configurations for this project
"""
from datetime import datetime

CHECKPOINT_PATH = 'checkpoint'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# tensorboard log dir
LOG_DIR = 'tensorboard'
