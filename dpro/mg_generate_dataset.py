''' Generate Dataset to train the Cost Model
'''
from tqdm import tqdm
import os,sys

from .arg_utils import SingleArg
from .logger_utils import SingleLogger

args = SingleArg().args
logger = SingleLogger(args.path.split(',')[0], 
    args.option, args.logging_level, 
    is_clean=args.clean, 
    show_progress=args.progress)
logger.info(args)

if args.option == "optimize":
    if args.sub_option == "train_amp":
        from cost_model._mixed_precision.amp_pred import AMPPredictor, train_amp_model
        train_amp_model()
        exit(0)
    elif args.sub_option == "train_gpu":
        from cost_model._gpu_predict.gpu_pred import train_gpu_model
        train_gpu_model()
        exit(0)