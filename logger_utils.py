import logging
import os
LOG_LEVEL_NAME = ["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]

#! define a singleton class
def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton

@Singleton
class SingleLogger:
    def __init__(self, path, name, logging_level, is_clean=False, show_progress=False):
        dirname = path if os.path.isdir(path) else os.path.dirname(path)
        logfile = os.path.join(dirname, "log_option-" + name + ".txt")
        if is_clean and os.path.exists(logfile):
            os.remove(logfile)
        #! config logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level=logging_level)

        #! bind some file stream
        handler = logging.FileHandler(logfile)
        handler.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if not show_progress: 
            #! if we want show progress, no need to bind the output stream 
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

    def info(self, msg, extra=None):
        self.logger.info(msg, extra=extra)

    def error(self, msg, extra=None):
        self.logger.error(msg, extra=extra)

    def debug(self, msg, extra=None):
        self.logger.debug(msg, extra=extra)

    def warn(self, msg, extra=None):
        self.logger.warn(msg, extra=extra)


