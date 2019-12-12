# Introduction

This project is used to analyze the trace results profiled via [byteprofile](https://github.com/joapolarbear/byteps) a developed version of [BytePS](https://github.com/bytedance/byteps).

# Usage
By choosing different `--option`, this project supports the functionalities as shown below.

## Statistic
Set arg `--option statistic` to show the statistic results, and arg `--path` must be set to the exact trace file path (ending with `.json` ).

## Visualize the DAG
Set arg `--option graph`, visualize the dependency dag graph. and arg `--path` must be set to the exact DAG path (ending with `.gml`).

## Combine Trace Files
Set arg `--option combine`, this can be used to combine several trace files into one file, e.g., one worker may has two GPUs, each of which generates a trace file, you can use this option and list the paths of these two files using `--path` and `--path2`.

There are two options to define the trace paths.

1. If you want to combine all the trace files on one worker, you can pass the trace directory path to `--path`, and the combined trace file will be stored under the same directory.
2. If you want to combine two specific trace files, you should use both `--path` and `--path2` to specify the exact trace file path. By default, the combined trace file will be store under the same directory as `--path`.

If you do not want combine all the traces, you can use `--filter` to give a list communication operations seperated with comma, then only these communication operations will appear in the combined trace file. For now, the filter only supports communication nodes.  An example is shown below.

```bash
python3 analyze.py --option combine --path ... --filter Comm.gradient_1,Comm.gradient_2
```


An example of combined timeline of 2 GPUs visualized by [chrome trace tool](chrome://tracing/) is shown below, which uses mnist as the dataset, running on 2 worker, each with 2 V100 GPUs. Here the prefix `Process 0`, `0` denotes the local rank of this GPU.

<img src="https://user-images.githubusercontent.com/17765864/68109805-764b5780-ff26-11e9-86ac-17d85394f8cf.png" width="720" height="440">

## Compare two trace files
Set arg `--option compare`.

## Calculate the Critical Path of the DAG
Set arg `--option critical`, here `--path` should be the root trace directory, by default, it's `BYTEPS_TRACE_DIR`. 

**Note that, you must use the latest version of byteprofile to run this option.**