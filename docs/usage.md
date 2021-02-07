
# Profiler
### Horovod + Tensorflow

* Global step should be created to use `hvd.TimelineHook`. and call
```opt = opt.minimize(loss, global_step=global_step)```
* `hvd.TimelineHook` generates smaller trace files.


---
# Optimizer

### Search both tensor fusion and operator fusion strategies

```
python3 analyze.py --option optimize --platform TENSORFLOW --comm_backend NCCL --nccl_algo RING --pretty --path /root/data/20210125_05_hvd_tf_resnet50_tcp/ --workspace /root/data/20210125_05_hvd_tf_resnet50_tcp/ --sub_option tensor_fusion,xla --xla_candidate_path /root/byteprofile-analysis/data/xla_candidates_resnet.txt
```