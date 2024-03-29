#!/bin/bash

sudo -i
cd ${HOME}/
rm -rf byteprofile-analysis
git clone https://github.com/joapolarbear/byteprofile-analysis.git
cd byteprofile-analysis
### install requirements
pip3 install -r requirements.txt

### Recompile XLA related Part or directly download it
# export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH} \
#     OLD_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH \
#     LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cudnn:/usr/local/cuda:/usr/local/cuda/compat:$OLD_LD_LIBRARY_PATH \
#     LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/nccl/lib/:$LIBRARY_PATH
# cd /root/tensorflow
# ./build_bpf_tf_modules.sh
cd ${HOME}/
wget https://github.com/joapolarbear/tensorflow/releases/download/v2.4.1-dev.2.0.1/dpro_xla_tools.zip
unzip dpro_xla_tools.zip

### Config env
# where the modified tensorflow locates
export BPF_TF_PATH=${HOME}/dpro_xla_tools
# the GPU id to run profiling on (specify one GPU only)
export BPF_COST_MODEL_PROFILE_GPU="0"
export CUDA_VISIBLE_DEVICES=0


export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib/python3.7/dist-packages/tensorflow/:$LD_LIBRARY_PATH
DRIVER_VERSION=$(nvidia-smi | grep -Po "CUDA Version: \K([0-9]{1,}\.)+[0-9]{1,}")
TOOLKIT_VERSION=$(nvcc --version | grep -Po "release \K([0-9]{1,}\.)+[0-9]{1,}")
echo "CUDA driver version: $DRIVER_VERSION"
echo "CUDA toolkit version: $TOOLKIT_VERSION"
### If the driver version is lower than the toolkit version, use compatability mode
# export LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH
sudo ln -sf /usr/local/lib/python3.7/dist-packages/tensorflow/libtensorflow_framework.so.2 /usr/lib/

export DPRO_GRAPHDEF_DFG_PATH=xxx

# The path where partition_def_0.json, tensor_shapes... are stored
TRACE_DIR=xxx
OUTPUT_DIR="${HOME}/xla"
mkdir -p $OUTPUT_DIR


### resnet
NUM_RANDOM_SAMPLES=5000
MAX_CLUSTER_SAMPLES=5
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=800

### VGG16
NUM_RANDOM_SAMPLES=5000
MAX_CLUSTER_SAMPLES=5
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=200

### VGG19
NUM_RANDOM_SAMPLES=2000
MAX_CLUSTER_SAMPLES=5
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=200

### InceptionV3
NUM_RANDOM_SAMPLES=5000
MAX_CLUSTER_SAMPLES=5
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=800

### generate data and train
cd ${HOME}/byteprofile-analysis
python3 xla_cm_entry.py --mode 0 \
    --trace_dir ${TRACE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_RANDOM_SAMPLES} \
    --max_cluster_samples ${MAX_CLUSTER_SAMPLES} \
    --min_cluster_size ${MIN_CLUSTER_SIZE} \
    --max_cluster_size ${MAX_CLUSTER_SIZE} \
    --batch_size 256

### exit root 
exit
if hdfs dfs -test -e /usr/hphu/xla_model/xla ; then
    hdfs dfs -rmr /usr/hphu/xla_model/xla
fi
hdfs dfs -put ${HOME}/xla /usr/hphu/xla_model/