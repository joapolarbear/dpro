# Introduction

This project is used to analyze the trace results profiled via [byteprofile](https://github.com/joapolarbear/byteps) a developed version of [BytePS](https://github.com/bytedance/byteps).

# Usage
By choosing different `--option`, this project supports the functionalities as shown below.

## Statistic
Set arg `--option statistic` to show the statistic results, and arg `--path` must be set to the exact trace file path (ending with `.json` ).

## Visualize the DAG
Set arg `--option graph`, visualize the dependency dag graph. and arg `--path` must be set to the exact DAG path (ending with `.gml`).

## Combine Trace Files
Set arg `--option combine`, this can be used to combine several trace files into one file, e.g., one worker may has two GPUs, each of which generates a trace file, you can use this option and list the paths of these two files using `--path`.

There are two options to define the trace paths.

1. Use file paths. In this case, `--path` should be a list of file paths, each of which denotes a trace file. The combined trace file will be stored under the same directory as the first trace file.
2. Use directory paths. In this case, `--path` is a list of directory paths, each of which denotes one worker and contains trace directories of GPUs on this worker. By default, the combined trace file will be stored under the first directory.

**Note: please ensure that all of paths are file paths or all of them are diretory paths.**


If you do not want combine all the traces, you can use `--filter` to give a list communication operations seperated with comma, then only these communication operations will appear in the combined trace file. For now, the filter only supports communication nodes.  An example is shown below.

```bash
python3 analyze.py --option combine --path ... --filter Comm.gradient_1,Comm.gradient_2
```


An example of combined timeline of 2 GPUs visualized by [chrome trace tool](chrome://tracing/) is shown below, which uses mnist as the dataset, running on 2 worker, each with 2 V100 GPUs. Here the prefix `Process 0`, `0` denotes the local rank of this GPU.

<img src="https://user-images.githubusercontent.com/17765864/68109805-764b5780-ff26-11e9-86ac-17d85394f8cf.png" width="720" height="440">

## Compare two trace files
Set arg `--option compare`. Similar to option `combine`, the argument `--path` could be a list of worker trace directories or a list of trace files. When a list of directories is given, traces on one worker will automatically be merged.

Besides, you can set `--xlsx` to export the comparison results to an XLSX file.

## Calculate the Critical Path of the DAG
Set arg `--option critical`, here `--path` should be the root trace directory, by default, it's `BYTEPS_TRACE_DIR`. 

**Note that, you must use the latest version of byteprofile to run this option.**

## Replay based on the traces
Set arg `--option reproduce` to reproduce the traces for one worker. 
Use `--path` to specify the path where the worker traces are stored, give `--del_queue` to include each partition and QueueType for communication traces.

## Update final traces
Set arg `--option collect` to update the final traces. In the meanwhile, the average iteration time would be outputed. `--path` should be the root directory of a worker or a GPU.
* --sub_option iter_time, only calculate the iteration time and FW+BW time
* --sub_option operator, update operator traces based on the source files.
* others, re-combine all traces based on the source files.

## `--option 3dcompare`
Ignore partition id

## ds
