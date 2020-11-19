# Format Specification

## Trace Format

```
### Uniform template
{
    "name": op_cat.name.sub_op,
    "ts": time_in_us,
    "dur": time_in_us,
    "pid": process_id, # e.g., host0.rank0
    "args": {
        "name": op_type.name.sub_op~>suffix,
        ...
        "cnt": 2, # how many times the op occurs,
        "step": 5, # which step the op is in, may be larger than "cnt"
    }
}
```

- `op_cat` is one of `BW`, `FW`, `Comm`, or specially, `trace["name"] = UPDATE_...`, e.g., `UPDATE_CAL`, `UPDATE_0`, `UPDATE_1`, ...  And `trace["name"] = I/O_...`
- For `Comm`, `sub_op` could be `SEND`, `REC`, and `suffix` could be `0_1_6_0` denotes `loopId`, `channelId`, `chunkId`, and `liceId` respectively

### For communication traces
Name should be tensor index `tensor_id` or `tensor_id_1+tensor_id_2+...+tensor_id_n` and the corresponding name should be stored in the `gradient_name_list` field in `<gpu_dir>/metadata.json`.

### Detailed communication traces
`"comm_detail"` in `ace["tid"]`


## Trace Statistic Format
```
name2sta = {
    op_long_name: {
        "avg": ...
        "cnt":
        "time":
        "min_t":
        "max_t":
        "id": ,# the order the op is created in the dict
    }
}

op_long_name = event["pid"]->event["name"] 
        or event["pid"]->event["name"]~>suffix
```

