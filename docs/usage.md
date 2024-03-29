
This script introduces how to analyze the traces of a distributed trianing job using dPRO toolkit. 
Suppose `DPRO_PATH` denotes the path where dpro is installed (check by entering `python3 -c "improt dpro; print(dpro.__path__)`), and `GLOBAL_TRACE_PATH` represents the path to store the final traces.

### Basic Usage
You can check the help information using the following commands
```
dpro_cli -h
dpro_cli --help
```

It will output all arguments of `DPRO_PATH/analyze.py`.  So you can use dPRO as a python script.
```
python3 DPRO_PATH/analyze.py <arguments>
```

It's tedious to set so many arguments everytime we want to analyze the traces of a job, so dPRO provides a command line tool, namely `dpro_cli`, which automatically search for a configuration file in YAML format (file extension .yaml). 
* The first argument `dpro_cli` must be `option`, which can be `collect`, `replay` or `optimize`. 
* The second argument `path` can be set as `GLOBAL_TRACE_PATH`. 
* Users can write a configuration file for each job. [sample_config.yaml](./sample_config.yaml) shows an example with three fields: 1) normal arguments; 2) `store_true` argument, which corresponds to the arguments set with `action=store_true`, see the help info for more details; 3) `env` field, which allow users set some environment variables of dPRO.
* `dpro_cli` will automaticall substitute `{{path}}` in the config file with the second argument `path`.

Writing a configuration file benefits when you want to fix some arguments for a job, e.g., platform (TENSORFLOW or MXNET), comm_backend (BYTEPS or NCCL), and so on. You can also add additional arguments for different analysis methods, e.g., sub_option, optimizer (DP or MCMC).


# Statistic
Users can statistic traces using the following commands.
```
python3 /home/tiger/byteprofile-analysis/analyze.py \
            --option collect \
            --platform TENSORFLOW \
            --comm_backend NCCL --nccl_algo RING --pretty \
            --path $GLOBAL_TRACE_PATH
```
or 
```
dpro_cli collect $GLOBAL_TRACE_PATH
```

# Replay
Users can simulate a training job using the following commands.
```
python3 /home/tiger/byteprofile-analysis/analyze.py \
            --option replay \
            --platform TENSORFLOW \
            --comm_backend NCCL --nccl_algo RING --pretty \
            --path $GLOBAL_TRACE_PATH
```
or
```
dpro_cli replay $GLOBAL_TRACE_PATH
```

---
# Optimizer

## Operator Fusion
### Search operator fusion strategies
Sample commands, put the XLA cost model in the path of  `./cost_model/_xla/.cost_model`
```
python3 analyze.py --option optimize --sub_option xla,^memory \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path $GLOBAL_TRACE_PATH --layer_by_layer --mcmc_beta 10 \
    --xla_candidate_path data/xla_candidates_resnet.txt
```
If you do not have a XLA cost model, run the following command to search with estimated fusion time:
```
python3 analyze.py --option optimize --sub_option xla \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty --simulate \
    --path $GLOBAL_TRACE_PATH \
    --workspace $GLOBAL_TRACE_PATH \
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
    --path $GLOBAL_TRACE_PATH \
    --workspace $GLOBAL_TRACE_PATH
```
or 
```
dpro_cli optimize $GLOBAL_TRACE_PATH --sub_option tsfs
```

## Combine Tensor Fusion and Operator Fusion
### Search both tensor fusion and operator fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option tensor_fusion,xla \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path $GLOBAL_TRACE_PATH \
    --workspace $GLOBAL_TRACE_PATH \
    --xla_candidate_path /root/byteprofile-analysis/data/xla_candidates_resnet.txt
```

### Generate tensor fusion strategies according to operator fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option from_opfs2tsfs \
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path $GLOBAL_TRACE_PATH,<cluster_mapping_path>
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
python3 mg_generate_dataset.py --option optimize --sub_option train_gpu --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --path $GLOBAL_TRACE_PATH
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


## Apply strategies

### Operator Fusion
We know XLA is enabled by setting `TF_XLA_FLAGS="--tf_xla_auto_jit=2"`, to apply customized XLA clustering strategies, set `XLA_CLUSTER_SPEC` to the path of the clustering specification file, where each row is in the format of `operator name` `cluster_id`

Besides, We can set `XLA_DUMP_DIR` to the path to store the intermediate informantion, which can be used to train the XLA cost model.
* `xla_candidates.txt`: candidates
* `unsafe_resource_deps.txt`: unsafe_resource_deps, __currently this file also contains xla_candidates.__
* `xla_clustering.txt`: the clustring strategies being applied, exits only when using default XLA (`XLA_CLUSTER_SPEC` is not set)

We can further set `TF_DUMP_GRAPH_PREFIX=${XLA_DUMP_DIR} TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2"` to dump graph_def.


### Tensor Fusion

#### BytePS
Use https://github.com/joapolarbear/byteps.git:byteprofile_rdma

You can also configure to fuse tensors to multiple tensor groups, by 
1. specifying the number of tensor groups by setting `BYTEPS_TENSOR_GROUP_NUM=x`
2. using a specification file by setting `BYTEPS_TENSOR_GROUP_FILE=/path/to/spec`. The file should be a json file and in the following format, where `0, 1, ...` denotes the indexes of tensors.
```
{
    "mapping": [
        "0+1+2",
        "3+4"
    ]
}
```


You can also configure the tensor partition size. A smaller size improves BytePS pipelining, but may have higher other overhead like NCCL coordination, ZMQ message headers, etc. The default and recommended value is 4096000 (in bytes).

```
export BYTEPS_PARTITION_BYTES=y
```

You can also configure the tensor partition size for each tensor use a specification file. Each line of the specification file should follow the format of `<tensor_name> <partition_size_in_bytes>`.
```
export BYTEPS_PARTITION_SPEC_FILE=/path/to/spec
```
Another way to specify the tensor partition size for a tensor is to use `BYTEPS_PARTITION_SPEC=<tensor_name>=<partition_size_in_bytes>`. You can also specify the tensor size for multiple tensors by seperating their specification with comma.

#### Horovod
Use https://github.com/joapolarbear/horovod:b_v0.21.0

You can also configure to fuse tensors to multiple tensor groups, by 
1. specifying the number of tensor groups by setting `HOROVOD_TENSOR_GROUP_NUM=x`
2. using a specification file by setting `HOROVOD_TENSOR_GROUP_FILE=/path/to/spec`. The file should be a json file and in the following format, where `0, 1, ...` denotes the indexes of tensors. Note that all tensor indexes should be specificed, even if one tensor is not fused with other tensors.
```
{
    "mapping": [
        "0+1+2",
        "3+4"
    ]
}
```
