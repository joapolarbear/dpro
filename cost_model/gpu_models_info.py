import numpy as np
ALL_GPU_MODELS = ["v100", "a100", "p100", "1080ti", "t4"]
CONFIG_NAMES = ["flops_fp32", "flops_fp16"]

### refer to https://www.microway.com/knowledge-center-articles/comparison-of-nvidia-geforce-gpus-and-nvidia-tesla-gpus/
### in tflops
GPU_CONFIG = np.array([
    [7.4, 29.7],
    [9.7, 78],
    [5, 19.95],
    [0.355, 0.177],
    [0.25, 16.2]
])
class GPUConfig:
    def __init__(self, gpu_model, configs):
        self.name = gpu_model
        self.flops_fp32 = configs[0]
        self.flops_fp16 = configs[1]

def ret_gpu_config(gpu_model):
    if gpu_model not in ALL_GPU_MODELS:
        raise ValueError("Invalid GPU Model name: {}".format(gpu_model))
    return GPUConfig(gpu_model, GPU_CONFIG[ALL_GPU_MODELS.index(gpu_model)])
