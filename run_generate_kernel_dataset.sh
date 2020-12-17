# where the modified tensorflow locates
export BPF_TF_PATH="/opt/tiger/tensorflow"
# the GPU id to run profiling on (specify one GPU only)
export BPF_COST_MODEL_PROFILE_GPU="0"

# modify these
TRACE_DIR="$BYTEPS_TRACE_DIR/0"
OUTPUT_DIR="/opt/tiger/xla/kernel_dataset"
NUM_RANDOM_SAMPLES=5000
MAX_CLUSTER_SAMPLES=5
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=800

python3 generate_kernel_dataset.py --trace_dir ${TRACE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_RANDOM_SAMPLES} \
    --max_cluster_samples ${MAX_CLUSTER_SAMPLES} \
    --min_cluster_size ${MIN_CLUSTER_SIZE} \
    --max_cluster_size ${MAX_CLUSTER_SIZE}