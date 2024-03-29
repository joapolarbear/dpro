# where the modified tensorflow locates
export BPF_TF_PATH="/root/tensorflow"
# this is the GPU used to compile XLA modules. Cost model will be run on another
# differnt GPU (specify one GPU here only)
export BPF_COST_MODEL_PROFILE_GPU="0"

# modify these
DATASET_DIR="/opt/tiger/xla/kernel_dataset"
OUTPUT_DIR="/opt/tiger/xla/cost_model"

python3 xla_train_module_cm.py --dataset_dir ${DATASET_DIR} --output_dir ${OUTPUT_DIR}
