import logging
import os
LOG_LEVEL_NAME = ["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]

def get_logger(args):
	path = args.path.split(',')[0]
	dirname = path if os.path.isdir(path) else os.path.dirname(path)
	logfile = os.path.join(dirname, "log_option-" + args.option + ".txt")
	if args.clean == True and os.path.exists(logfile):
		os.remove(logfile)
	#! config logging
	logger = logging.getLogger(__name__)
	logger.setLevel(level=args.logging_level)

	handler = logging.FileHandler(logfile)
	handler.setLevel(args.logging_level)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(formatter)
	logger.addHandler(console)

	return logger
