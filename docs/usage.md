
# Profiler
### Horovod + Tensorflow

* Global step should be created to use `hvd.TimelineHook`. and call
```opt = opt.minimize(loss, global_step=global_step)```
* `hvd.TimelineHook` generates smaller trace files.


---
# Optimizer

### Search tensor fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option tensor_fusion 
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/
```

### Search both tensor fusion and operator fusion strategies
Sample commands
```
python3 analyze.py --option optimize --sub_option tensor_fusion,xla 
    --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty \
    --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/ \
    --xla_candidate_path /root/byteprofile-analysis/data/xla_candidates_resnet.txt
```