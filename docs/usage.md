
# Profiler
### Horovod + Tensorflow

* Global step should be created to use `hvd.TimelineHook`. and call
```opt = opt.minimize(loss, global_step=global_step)```
* `hvd.TimelineHook` generates smaller trace files.

# Replay
```
python3 /home/tiger/byteprofile-analysis/analyze.py \
            --option replay \
            --platform TENSORFLOW \
            --comm_backend NCCL --nccl_algo RING --pretty \
            --path $GLOBAL_TRACE_PATH
```

---
# Optimizer
### Search operator fusion strategies
Sample commands, put the XLA cost model in the path of  `./cost_model_xla/.cost_model`
```
python3 analyze.py --option optimize --sub_option xla,^memory \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --xla_candidate_path data/xla_candidates_resnet.txt \
    --update_infi_para
```
If you do not have a XLA cost model, run the following command to search with simulation
```
python3 analyze.py --option optimize --sub_option xla \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty --simulate \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --xla_candidate_path data/xla_candidates_resnet.txt \
    --update_infi_para
```

### Search tensor fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option tensor_fusion \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/
```

### Search both tensor fusion and operator fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option tensor_fusion,xla \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --xla_candidate_path /root/byteprofile-analysis/data/xla_candidates_resnet.txt
```

### Generate tensor fusion strategies according to operator fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option from_opfs2tsfs \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/,<cluster_mapping_path>
```
where `<cluster_mapping_path>` denotes the path to the cluster_mapping.txt (operator fusion search result).

### Mixed Precision Training
`TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_PRIORLIST_FILE`: a file containing ops to force quantize, seperated by \n
`TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_PRIORLIST_ADD`: ops to force quantize, seperated by comma
`TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_FORCE`: clear the CLEARLIST and BCACKLIST if set

## Train Cost Model
### Cost Model for MultiGPU

```
python3 mg_generate_dataset.py --option optimize --sub_option train_gpu --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --path /path/to/traces
```
`--path` specifies where traces are stored, organized by the GPU model name and ML model name
