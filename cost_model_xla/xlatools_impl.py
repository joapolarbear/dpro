import ctypes
from pathlib import Path
import subprocess
import re

_XLATOOLS_CLIB_PATH = "/root/tensorflow/bazel-bin/tensorflow/compiler/byteprofile_xlatools/byteprofile_xlatools_c_api.so"

_XLATOOLS_CLIB = ctypes.CDLL(_XLATOOLS_CLIB_PATH)

_XLATOOLS_CLIB.C_API_CompileToHlo.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_XLATOOLS_CLIB.C_API_GenFeatureVector.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float]

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

def compile_to_hlo(graph_path, config_path, dump_path_unopt, dump_path_opt):
    _check_arg_types([graph_path, config_path, dump_path_unopt, dump_path_opt], [str] * 4)
    _check_file_exist_for_reading(graph_path)
    _check_file_exist_for_reading(config_path)
    _check_file_available_for_writing(dump_path_unopt)
    _check_file_available_for_writing(dump_path_opt)
    status = _XLATOOLS_CLIB.C_API_CompileToHlo(graph_path.encode(), config_path.encode(), dump_path_unopt.encode(), dump_path_opt.encode())
    if status != 0:
        print(status)
        raise RuntimeError("Failed to compile to HLO.")

def gen_feature_vector(hlo_module_path, output_path, gflops_per_second, gbytes_per_second):
    _check_arg_types([hlo_module_path, output_path, gflops_per_second, gbytes_per_second], [str, str, float, float])
    _check_file_exist_for_reading(hlo_module_path)
    _check_file_available_for_writing(output_path)
    status = _XLATOOLS_CLIB.C_API_GenFeatureVector(hlo_module_path.encode(), output_path.encode(), gflops_per_second, gbytes_per_second)
    if status != 0:
        raise RuntimeError("Failed to generate feature vector.")

def replay_hlo(hlo_path, replay_exec=None):
    if replay_exec is None:
        replay_exec = "/root/tensorflow/bazel-bin/tensorflow/compiler/xla/tools/replay_computation_gpu"
    opt_1 = "--num_runs=30"
    opt_2 = "--use_fake_data"
    process = subprocess.run([replay_exec, opt_1, opt_2, hlo_path], capture_output=True)
    output = process.stderr.decode("ascii")
    times = [float(line.split()[3][:-2]) for line in re.findall("Done executing in .*s:", output)]
    return sum(times) / len(times)