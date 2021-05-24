# Format Specification

## Trace Format

```python
### Uniform template
{
    "name": op_cat.op_name.sub_op,
    "ts": time_in_us,
    "dur": time_in_us,
    "pid": process_id, # e.g., host0.rank0
    "args": {
        "name": op_type.op_name.sub_op~>suffix,
        ...
        "cnt": 2, # how many times the op occurs,
        "step": 5, # which step the op is in, may be larger than "cnt"
    }
}
```


- `op_cat` is one of `BW`, `FW`, `Comm`, `UPDATE_`, or specially, `trace["name"] = UPDATE_...`, e.g., `UPDATE_CAL`, `UPDATE_0`, `UPDATE_1`, ...  And `trace["name"] = I/O_...`
- `op_name` is the raw name profiled by the built-in profiler of each ML framework.
- For `Comm`, `sub_op` could be `SEND`, `REC`, and `suffix` could be `0_1_6_0` denotes `loopId`, `channelId`, `chunkId`, and `liceId` respectively
  
We call the names follow the format of `op_cat.op_name.sub_op` as `standard name`

### For communication traces
Name should be tensor index `tensor_id` or `tensor_id_1+tensor_id_2+...+tensor_id_n` and the corresponding `tensor_name` should be stored in the `gradient_name_list` field in `<gpu_dir>/metadata.json`.

### Detailed communication traces
`"comm_detail"` in `trace["tid"]`


## Trace Statistic Format
``` python
name2sta = {
    op_long_name: {
        "avg": ...
        "cnt":
        "time":
        "min_t":
        "max_t":
        "id": ,# the order the op is created in the dict
        "step_ids": [] # a list of index, denotes where this operator appears in the traces
}

op_long_name = event["pid"]->event["name"] 
        or event["pid"]->event["name"]~>suffix
```

## Dependency Graph
Nodes: 
```python
op_long_name: {
    "avg": time_in_us,
    gap_string:time_in_us
}
```
`gap_string` denotes different kinds of gaps

Special Nodes `END`, the end node.

## NCCL Graph
- During trace collection, NCCL graph needs to parse at least one GPU's NCCL traces to get `chunkNum`, `sliceNum`, `channelNum`, `loopNum` for each `raw_name` (`op_cat.op_name`, without `sub_op`)
- During trace collection, we need to parse `nccl_rank_graph.json` to get the connection information of this GPU.

## ParameterDict
Manage the parameter info of a DNN model. Seek to implement a unified `ParameterDict`, but now, it is only for MXNet.

### MXNet
Contains:
- `gradient_name_list`, which maps `tensor_id` to `tensor_name`;
- `tensor2update`, which maps `tensor_id` to `update_id`


## Rules of converting Framework traces
### Tensorflow
#### UPADATE operators
1. Take all down stream operators of `Comm` as UPDATE operators
2. There may be depedency between UPDATE operators
#### FW and BW operators
1. **Assumption**: in TensorFlow, some operators may have multiple traces with the same name in one step, which we call as sub_trace, we assume they are continuous and combine them into one single operator.

#### Statistic the number of step
1. pre_cat not in [io, fw], cur_cat in [io, fw], step cnt + 1

### MXNET
#### UPDATE operators
1. We assume there is no dependency between UPDATE operators, except for `UPDATE_CAL`->`UPDATE_ID`
