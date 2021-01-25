# where the modified tensorflow locates
export BPF_TF_PATH="/root/tensorflow"
# this is the GPU used to compile XLA modules. Cost model will be run on another
# differnt GPU (specify one GPU here only)
export BPF_COST_MODEL_PROFILE_GPU="0"

# modify these
DATASET_DIR="/PATH/TO/DATASET/DIR"
COST_MODEL_DIR="/PATH/TO/COST/MODEL"

python3 xla_test_module_cm.py --dataset_dir ${DATASET_DIR} --cost_model_dir ${COST_MODEL_DIR}