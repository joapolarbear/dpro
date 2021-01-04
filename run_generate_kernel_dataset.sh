#!/bin/bash
cd /opt/tiger
if hdfs dfs -test -e /usr/hphu/xla_model ; then
    hdfs dfs -get /usr/hphu/xla_model/xla
    hdfs dfs -get /usr/hphu/xla_model/traces
fi

sudo -i
cd /opt/tiger/
rm -rf byteprofile-analysis
git clone https://github.com/joapolarbear/byteprofile-analysis.git


export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH} \
    OLD_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cudnn:/usr/local/cuda:/usr/local/cuda/compat:$OLD_LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/nccl/lib/:$LIBRARY_PATH

cd /opt/tensorflow
./build...

## Set enviroment variables related to profiling
export BYTEPS_TRACE_ON="${BYTEPS_TRACE_ON:-1}"
if [ "${BYTEPS_TRACE_ON}" = "1" ]; then
	export BYTEPS_TRACE_DIR="/opt/tiger/traces"
	export BYTEPS_TRACE_START_STEP="${BYTEPS_TRACE_START_STEP:-50}"
	export BYTEPS_TRACE_END_STEP="${BYTEPS_TRACE_END_STEP:-60}"
    echo "BYTEPS_TRACE_DIR: ${BYTEPS_TRACE_DIR}"
    echo "BYTEPS_TRACE_START_STEP: ${BYTEPS_TRACE_START_STEP}"
    echo "BYTEPS_TRACE_END_STEP: ${BYTEPS_TRACE_END_STEP}"
fi
# where the modified tensorflow locates
export BPF_TF_PATH="/root/tensorflow"
# the GPU id to run profiling on (specify one GPU only)
export BPF_COST_MODEL_PROFILE_GPU="0"

# modify these
TRACE_DIR="$BYTEPS_TRACE_DIR/0"
OUTPUT_DIR="/opt/tiger/xla/kernel_dataset"
NUM_RANDOM_SAMPLES=2000
MAX_CLUSTER_SAMPLES=3
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=800


cd /opt/tiger/byteprofile-analysis
python3 generate_kernel_dataset.py --trace_dir ${TRACE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_RANDOM_SAMPLES} \
    --max_cluster_samples ${MAX_CLUSTER_SAMPLES} \
    --min_cluster_size ${MIN_CLUSTER_SIZE} \
    --max_cluster_size ${MAX_CLUSTER_SIZE}

# modify these
DATASET_DIR="/opt/tiger/xla/kernel_dataset"
OUTPUT_DIR="/opt/tiger/xla/cost_model"

cp /opt/tiger/xla/kernel_dataset/cleaned_graph.json /opt/tiger/xla/kernel_dataset/dataset/
cp /opt/tiger/xla/kernel_dataset/tensor_shapes.json /opt/tiger/xla/kernel_dataset/dataset/

cd /opt/tiger/byteprofile-analysis
python3 train_module_cost_model.py --dataset_dir ${DATASET_DIR} --output_dir ${OUTPUT_DIR}

if hdfs dfs -test -e /usr/hphu/xla_model/xla ; then
    hdfs dfs -rmr /usr/hphu/xla_model/xla
fi
hdfs dfs -put /opt/tiger/xla /usr/hphu/xla_model/