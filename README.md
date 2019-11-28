# Introduction

This project is used to analyze the trace results profiled via [byteprofile](https://github.com/joapolarbear/byteps) a developed version of [BytePS](https://github.com/bytedance/byteps).

# Usage
By choosing different `--option`, this project supports the functionalities as shown below.

## Statistic
Set arg `--option statistic` to show the statistic results, and arg `--path` must be set to the exact trace file path (ending with `.json` ).

## Visualize the DAG
Set arg `--option graph`, visualize the dependency dag graph. and arg `--path` must be set to the exact DAG path (ending with `.gml`).

## Combine two trace files
Set arg `--option combine`, this can be used to combine two trace files into one file, e.g., one worker may has two GPUs, each of which generates a trace file, you can use this option and list the paths of these two files using `--path` and `--path2`


An example of combined timeline of 2 GPUs visualized by [chrome trace tool](chrome://tracing/) is shown below, which uses mnist as the dataset, running on 2 worker, each with 2 V100 GPUs. Here the prefix `Process 0`, `0` denotes the local rank of this GPU.

<img src="https://user-images.githubusercontent.com/17765864/68109805-764b5780-ff26-11e9-86ac-17d85394f8cf.png" width="720" height="440">

## Compare two trace files
Set arg `--option compare`, 

## Calculate the Critical Path of the DAG
Set arg `--option critical`, here `--path` should be the root trace directory, by default, it's `BYTEPS_TRACE_DIR`. 

**Note that, you must use the latest version of byteprofile to run this option.**