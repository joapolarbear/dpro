from . import base
from . import logger_utils
from . import collect
from . import trace_utils
from . import replay

def init(workspace, name, **kwargs):
    from dpro.logger_utils import SingleLogger
    logger = SingleLogger(workspace, name, **kwargs)