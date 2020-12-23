import ctypes
from pathlib import Path
import subprocess
import os
import re

from cost_model_xla.constant_utils import *

if CMEnvs.TF_PATH in os.environ:
    BPF_TF_PREFIX = os.environ[CMEnvs.TF_PATH]
else:
    BPF_TF_PREFIX = None
    print("[WARNING] Environment {} not set. Guessing default TF location.".format(CMEnvs.TF_PATH))

if CMEnvs.CM_PROFILE_GPU in os.environ:
    try:
        BPF_PROFILE_GPU = int(os.environ[CMEnvs.CM_PROFILE_GPU])
    except:
        print("[ERROR] Invalid {} value (must be an integer).".format(CMEnvs.CM_PROFILE_GPU))
        exit(-1)
else:
    print("[ERROR] Required environment {} value not set.".format(CMEnvs.CM_PROFILE_GPU))
    exit(-1)

def _check_file_available_for_writing(path):
    p = Path(path)
    p_dir = p.resolve().parent
    if not p_dir.is_dir():
        p.mkdir(parents=True)

def _check_file_exist_for_reading(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("Cannot find file {}".format(path))

def _check_arg_types(args, types):
    if len(args) != len(types):
        raise RuntimeError("Mismatch number of arguments and types in _check_arg_types. ({} v.s. {})".format(len(args), len(types)))
    for index, (arg, arg_type) in enumerate(zip(args, types)):
        if not isinstance(arg, arg_type):
            raise TypeError("Inappropriate argument type for argument {}. Expected {} but got {}".format(index, arg_type, type(arg)))

def compile_to_hlo(graph_path, config_path, dump_path_unopt, dump_path_opt, compile_exec=None):
    if compile_exec is None:
        if BPF_TF_PREFIX is not None:
            compile_exec = os.path.join(BPF_TF_PREFIX, "bazel-bin/tensorflow/compiler/byteprofile_xlatools/tfcompile_hlo")
        else:
            compile_exec = "/root/tensorflow/bazel-bin/tensorflow/compiler/byteprofile_xlatools/tfcompile_hlo"
    if not os.path.exists(compile_exec):
        print("Cannot find the path to replay_computation_gpu.")
        exit(-1)

    _check_arg_types([graph_path, config_path, dump_path_unopt, dump_path_opt], [str] * 4)
    _check_file_exist_for_reading(graph_path)
    _check_file_exist_for_reading(config_path)
    _check_file_available_for_writing(dump_path_unopt)
    _check_file_available_for_writing(dump_path_opt)
    subprocess.run("CUDA_VISIBLE_DEVICES={} {} {} {} {} {}".format(str(BPF_PROFILE_GPU), compile_exec, graph_path, config_path, dump_path_unopt, dump_path_opt), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, shell=True)

def replay_and_generate_kernel_sample(sample_id_start, hlo_path, tmp_dir, dataset_path, replay_exec=None):
    if replay_exec is None:
        if BPF_TF_PREFIX is not None:
            replay_exec = os.path.join(BPF_TF_PREFIX, "bazel-bin/tensorflow/compiler/xla/tools/replay_computation_gpu")
        else:
            replay_exec = "/root/tensorflow/bazel-bin/tensorflow/compiler/xla/tools/replay_computation_gpu"
    if not os.path.exists(replay_exec):
        print("Cannot find the path to replay_computation_gpu.")
        exit(-1)
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(BPF_PROFILE_GPU)
    opt_1 = "--num_runs=50"
    opt_2 = "--use_fake_data=true"
    opt_3 = "--print_result=false"
    opt_4 = "--dataset_path={}".format(dataset_path)
    opt_5 = "--temp_dir_path={}".format(tmp_dir)
    opt_6 = "--profile_start=30"
    opt_7 = "--profile_end=50"
    opt_8 = "--sample_id_start={}".format(sample_id_start)
    process = subprocess.run("CUDA_VISIBLE_DEVICES={} {} {} {} {} {} {} {} {} {} {}".format(str(BPF_PROFILE_GPU), replay_exec, opt_1, opt_2, opt_3, opt_4, opt_5, opt_6, opt_7, opt_8, hlo_path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=my_env, shell=True, check=True)

def extract_kernel_features_from_hlo(hlo_path, tmp_dir, extract_exec=None):
    if extract_exec is None:
        if BPF_TF_PREFIX is not None:
            extract_exec = os.path.join(BPF_TF_PREFIX, "bazel-bin/tensorflow/compiler/xla/tools/extract_features_from_hlo")
        else:
            extract_exec = "/root/tensorflow/bazel-bin/tensorflow/compiler/xla/tools/extract_features_from_hlo"
    if not os.path.exists(extract_exec):
        print("Cannot find the path to replay_computation_gpu.")
        exit(-1)
        
    opt_1 = "--hlo_path={}".format(hlo_path)
    opt_2 = "--temp_dir_path={}".format(tmp_dir)
    subprocess.run([extract_exec, opt_1, opt_2], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)