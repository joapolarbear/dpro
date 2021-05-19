
# Profiler
### Horovod + Tensorflow
```
recorder = hvd.Recorder(model=model, batch_size=args.batch_size)

@hvd.profile(recorder)
@tf.function
def benchmark_step(first_batch):
    ...
    with tf.GradientTape() as tape:
        ...
    tape = hvd.DistributedGradientTape(tape)
    ...
```

# Statistic
```
python3 /home/tiger/byteprofile-analysis/analyze.py \
            --option statistic \
            --platform TENSORFLOW \
            --comm_backend NCCL --nccl_algo RING --pretty \
            --path $GLOBAL_TRACE_PATH \
            --update_infi_para
```

# Replay
```
python3 /home/tiger/byteprofile-analysis/analyze.py \
            --option replay \
            --platform TENSORFLOW \
            --comm_backend NCCL --nccl_algo RING --pretty \
            --path $GLOBAL_TRACE_PATH \
            --update_infi_para
```

---
# Optimizer

## Operator Fusion
### Search operator fusion strategies
Sample commands, put the XLA cost model in the path of  `./cost_model/_xla/.cost_model`
```
python3 analyze.py --option optimize --sub_option xla,^memory \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --xla_candidate_path data/xla_candidates_resnet.txt \
    --update_infi_para --layer_by_layer --mcmc_beta 10
```
If you do not have a XLA cost model, run the following command to search with estimated fusion time:
```
python3 analyze.py --option optimize --sub_option xla \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty --simulate \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --xla_candidate_path data/xla_candidates_resnet.txt \
    --update_infi_para --layer_by_layer
```

### Sample some example strategies
Fuse operators layer by layer, below is an exmple where each 2 layers' operators are fused.
```
python3 analyze.py --option optimize --sub_option xla,^memory \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty  \
    --path path/to/trace/directory \
    --xla_candidate_path path/to/candidate/file/ \
    --update_infi_para --simulate --layer_num_limit 2
```

## Tensor Fusion
### Search tensor fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option tensor_fusion \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/
```

## Combine Tensor Fusion and Operator Fusion
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


## Mixed Precision Training
`TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_PRIORLIST_FILE`: a file containing ops to force quantize, seperated by \n
`TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_PRIORLIST_ADD`: ops to force quantize, seperated by comma
`TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_FORCE`: clear the CLEARLIST and BCACKLIST if set

---
# Train Cost Model
## Cost Model for MultiGPU

```
python3 mg_generate_dataset.py --option optimize --sub_option train_gpu --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --path /path/to/traces
```
`--path` specifies where traces are stored, organized by the GPU model name and ML model name

---

### Heat-based Search Algorithm for Operator Fusion
#### Requirements of Weights
1. Initially, each strategy has a weight of 1
2. If fuse(a, b) generating c brings speedup,
  1. The weights of fusion strategies involving a, b, c > 1
  2. The weights of de-fusion strategies involving a, b, c < 1
3. If fuse(a, b) generating c leads to worse performance,
  3. The weights of fusion strategies involving a, b, c < 1
  4. The weights of de-fusion strategies involving a, b, c > 1
4. If defuse(c) generating a, b is better, the same as item 3
5. If defuse(c) generating a, b is worse, the same as item 2
6. 
#### Solution
- The heat is directional, i.e., a large heat means an operator is expected to participate in operator fusion, but not in operator partition
  - After applying a strategy, if it's a fusion strategy, record  Delta T at the heat history list, otherwise, ecord  - Delta T at the heat history list. 
  - To calculate the final heat H of one operator, if the heat history list is empty, return 0, otherwise, return ..., k > 1, thus H > -1
        $$H = \frac{1}{n}\sum_i^{n}\frac{e^{\Delta T_i} - 1}{k \Delta t_i}$$
  - With the heat H, calculate the final weight W as follows
        $$W = \left\{
        \begin{array}{rcl}
        1 + H & &, fusion \quad strategy\\
        1  + \frac{1}{H+1} - 1 = \frac{1}{H+1} & & , partition \quad strategy\\
        \end{array} \right.$$