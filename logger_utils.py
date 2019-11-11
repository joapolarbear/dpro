import logging
import os
LOG_LEVEL_NAME = ["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]

def get_logger(args):
	#! config logging
	logger = logging.getLogger(__name__)
	logger.setLevel(level=args.logging_level)

	handler = logging.FileHandler(os.path.join(os.path.dirname(args.path), args.option + "_" + args.logging_file))
	handler.setLevel(args.logging_level)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(formatter)
	logger.addHandler(console)

	return logger
