# Profiler
### Horovod + Tensorflow
Please follow the instructions in [docs/dependency.md](./dependency.md) to install our customized Horovod and TensorFlow.
To enable profiling, add the following code to your script for a training job using Horovod.

```
recorder = hvd.Recorder()

@hvd.profile(recorder)
@tf.function
def benchmark_step(first_batch):
    ...
    with tf.GradientTape() as tape:
        ...
    tape = hvd.DistributedGradientTape(tape)
    ...
```

Besides, the following environment variables needs to be set
```
export BYTEPS_TRACE_ON=1
export BYTEPS_TRACE_DIR=path/to/store/traces
export BYTEPS_TRACE_START_STEP=<profiling_start_steps>
export BYTEPS_TRACE_END_STEP==<profiling_end_steps>
```
Then, launch the distributed trianing job on a cluster.


Before analyzing traces using the dPRO toolkit, you need to collect traces from different workers to one device and organize them in the following manner. 
```
global_traces/
    |
    - host0/        # traces of device 0
        |
        - 0/        # traces of GPU 0 on device 0
        |
        - 1/        # traces of GPU 1 on device 0
        ...
    |
    - host1/ 
        |
        ...
```