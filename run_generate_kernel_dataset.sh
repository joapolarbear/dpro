# where the modified tensorflow locates
export BPF_TF_PATH="/root/tensorflow"
# the GPU id to run profiling on (specify one GPU only)
export BPF_PROFILE_GPU="0"

# modify these
TRACE_DIR="/PATH/TO/GPU/LEVEL/TRACE/DIR"
OUTPUT_DIR="/PATH/TO/OUTPUT/DATASET"
NUM_RANDOM_SAMPLES=4000
MAX_CLUSTER_SAMPLES=3
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=800

python3 generate_kernel_dataset.py --trace_dir ${TRACE_DIR} --output_dir ${OUTPUT_DIR} --num_samples ${NUM_RANDOM_SAMPLES} --max_cluster_samples ${MAX_CLUSTER_SAMPLES} --min_cluster_size ${MIN_CLUSTER_SIZE} --max_cluster_size ${MAX_CLUSTER_SIZE}