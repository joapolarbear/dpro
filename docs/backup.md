# Commands for dPRO 
We have `bash setup.sh` to install dPRO now, the following commands are for archive

## Install dPRO
`
cd ${HOME}/
rm -rf dpro
git clone https://github.com/joapolarbear/dpro.git
cd dpro && sudo bash setup.sh
`
## debug mode
`
pip3 install -e $HOME/ws/git/dpro
`
—--

## Reinstall customized TF
```
wget https://github.com/joapolarbear/tensorflow/releases/download/v2.4.1-dev.2.0.2/tensorflow-2.4.1-cp37-cp37m-linux_x86_64.whl
pip3 --no-cache-dir install --force-reinstall tensorflow-2.4.1-cp37-cp37m-linux_x86_64.whl
```

## RUN
```
export HOROVOD_FUSION_THRESHOLD="${HOROVOD_FUSION_THRESHOLD:-67108864}"
export HOROVOD_CYCLE_TIME="${HOROVOD_CYCLE_TIME:-0}"
export HOROVOD_LOG_LEVEL="${HOROVOD_LOG_LEVEL:-warning}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"

export HOROVOD_FUSION_THRESHOLD=0
export HOROVOD_CYCLE_TIME=5

bash mpirun.sh python3 $HOME/horovod_examples/tensorflow/tensorflow_synthetic_benchmark.py --model VGG16 --num-iters 5

bash mpirun.sh nsys profile -o 1ib_overlap_xlaoff_gpu%q{OMPI_COMM_WORLD_RANK}.qdrep python3 $HOME/horovod_examples/tensorflow/tensorflow_synthetic_benchmark.py --model VGG16 --num-iters 5

TF_XLA_FLAGS=--tf_xla_auto_jit=2

for (( id=0; id < 8; id++ )); do
    python3 $HOME/nvprof2json/nvprof2json.py --filename $HOME/global_traces/host0/simple.${id}.nvprof --filter CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL > $HOME/global_traces/rank${id}.json
done
for (( id=8; id < 16; id++ )); do
    python3 $HOME/nvprof2json/nvprof2json.py --filename $HOME/global_traces/host1/simple.${id}.nvprof --filter CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL > $HOME/global_traces/rank${id}.json
done
```

---
## Train xla cost model
```
cd ${HOME}/
rm -rf dpro
git clone https://github.com/joapolarbear/dpro.git
cd dpro &&  sudo bash setup.sh

cd ${HOME}/
wget https://github.com/joapolarbear/tensorflow/releases/download/v2.4.1-dev.2.0.1/dpro_xla_tools.zip
unzip dpro_xla_tools.zip
export BPF_TF_PATH=${HOME}/dpro_xla_tools
sudo ln -sf /usr/local/lib/python3.7/dist-packages/tensorflow/libtensorflow_framework.so.2 /usr/lib/
```

## The GPU id to run profiling on (specify one GPU only)
```
export BPF_COST_MODEL_PROFILE_GPU="0"
export CUDA_VISIBLE_DEVICES=0

cd ${HOME}
COMM_BACKEND_LAUNCHER="python3 /usr/local/byteps/launcher/launch.py python3 test.py --comm_backend bps"
COMM_BACKEND_LAUNCHER="horovod -np 1 python3 test.py"
```
### RUN
```
$COMM_BACKEND_LAUNCHER
ALL_TRACE_DIR=${HOME}/trace_dirs_vgg16
mv $HOME/traces  $ALL_TRACE_DIR

export XLA_DUMP_DIR=${HOME}/xla_dump
mkdir -p $XLA_DUMP_DIR
TF_DUMP_GRAPH_PREFIX=${XLA_DUMP_DIR} TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" $COMM_BACKEND_LAUNCHER

export DPRO_GRAPHDEF_DFG_PATH=${XLA_DUMP_DIR}/graphdef_dag.gml
export TRACE_DIR=$ALL_TRACE_DIR/0
export OUTPUT_DIR="${HOME}/xla_vgg16"
mkdir -p $OUTPUT_DIR

NUM_RANDOM_SAMPLES=5000
MAX_CLUSTER_SAMPLES=5
MIN_CLUSTER_SIZE=4
MAX_CLUSTER_SIZE=800

cd ${HOME}/dpro
python3 xla_cm_entry.py --mode 0 \
--trace_dir ${TRACE_DIR} \
--output_dir ${OUTPUT_DIR} \
--num_samples ${NUM_RANDOM_SAMPLES} \
--max_cluster_samples ${MAX_CLUSTER_SAMPLES} \
--min_cluster_size ${MIN_CLUSTER_SIZE} \
--max_cluster_size ${MAX_CLUSTER_SIZE} \
--batch_size 256

```
## TEST the searched results
```
hdfs dfs -rm -r /usr/hphu/search_rst && hdfs dfs -mkdir /usr/hphu/search_rst

function put_spec_to_hdfs {
	hdfs dfs -put $1/spec /usr/hphu/search_rst/$1_spec
}

put_spec_to_hdfs 20210929_01_bps_tf_resnet50_tcp_2w8g2s_tsfs_tspart_optws

hdfs dfs -ls /usr/hphu/search_rst
```


# BytePS

## 重装byteps
```
cd /usr/local/byteps && git pull && git submodule update
cd /usr/local/byteps/3rdparty/ps-lite && make clean && make -j USE_RDMA=1 && \
cd /usr/local/byteps/ && rm -rf build && \
BYTEPS_USE_RDMA=1 BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_MXNET=1 python3 setup.py install
```
### Test byteps
```
export DMLC_ROLE=scheduler
export DMLC_ROLE=worker
export DMLC_WORKER_ID=0

export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.129.120.196
export DMLC_PS_ROOT_PORT=8008

unset NCCL_ALGO
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG
unset NCCL_TRACE_START_STEP
unset NCCL_TRACE_DIR
unset NCCL_TRACE_END_STEP
unset NCCL_ENABLE_TIMELINE
export BYTEPS_LOG_LEVEL=INFO

cd $HOME/bert && sudo git checkout b_tf2_4
python3 /usr/local/byteps/launcher/launch.py python3 $HOME/bert/run_pretraining.py
```

# NCCL Contention 测试
```
rm -rf /usr/local/nccl
pip3 uninstall -y horovod

cd /usr/local && git clone https://github.com/NVIDIA/nccl.git
cd /usr/local/nccl && git checkout v2.10.3-1
rm -rf /usr/include/nccl.h

make -j src.build && make pkg.txz.build 
tar -Jxf ./build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1
echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
ldconfig && ln -sf /usr/local/nccl/include/* /usr/include/

HOROVOD_NCCL_HOME=/usr/local/nccl \
HOROVOD_NCCL_HOME=/usr/local/nccl \
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL \
HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 \
pip3 install --no-cache-dir horovod==0.21.0
```

