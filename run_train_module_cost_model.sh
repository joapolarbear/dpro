# where the modified tensorflow locates
export BPF_TF_PATH="/root/tensorflow"
# this is the GPU used to compile XLA modules. Cost model will be run on another
# differnt GPU (specify one GPU here only)
export BPF_COST_MODEL_PROFILE_GPU="0"

# modify these
DATASET_DIR="/PATH/TO/DATASET/DIR"
OUTPUT_DIR="/PATH/TO/OUTPUT/COST/MODEL"

python3 train_module_cost_model.py --dataset_dir ${DATASET_DIR} --output_dir ${OUTPUT_DIR}