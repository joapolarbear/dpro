export BPF_TF_PATH="/root/tensorflow"
export BPF_PROFILE_GPU="0"

python3 generate_kernel_dataset.py --trace_dir /root/capture_file/run_0_dec8/traces_0/0 --output_dir /root/dataset_mixed_dec8_0 --num_samples 10 --max_cluster_samples 3 --min_cluster_size 4 --max_cluster_size 800