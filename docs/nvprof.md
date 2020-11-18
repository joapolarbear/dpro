# Introduction

This tutorial gives an introduction of how to compare ByteProfile Traces and NvProf traces

# How to use

## Traces Collection

Use `nvprof` to collect NvProf traces.
```nvprof -o foo.nvvp your_program```

In the meanwhile, you should customized the source code and enable BPF_related environment variabless correspondingly to collect ByteProfile Traces. 

Then you can run `python3 nvprof2json.py --filename <nvprof_path> > <target_path>` to convert NvProf Traces from `.nvvp` to `JSON` format.

## Comparison

For comparison, you can simply run
`python3 analyze.py --option mapping --path <ByteProfile_path>,<nvprof_json_path>`
Then the statitical results will be stored in `mapfrom_op2kernels.xlsx` under the same folder as `<nvprof_json_path>`

The rational is
1. for the first iteration, check the bias. check those relatively large kernel-level traces, it overlapping with a op-level trace but is not convered by that
2. for the second iteration, generate the mapping