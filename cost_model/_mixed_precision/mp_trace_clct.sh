#!/bin/bash
set -e
### MXNet env
MXNET_CUDNN_AUTOTUNE_DEFAULT=0
MXNET_GPU_WORKER_NTHREADS=1
MXNET_EXEC_BULK_EXEC_TRAIN=0

### Profiling env
# BYTEPS_TRACE_ON=1 
# BYTEPS_TRACE_DIR=/opt/tiger/traces

MODEL='ResNet50'
PLATFORM='TF'
echo "Platform: ${PLATFORM}, Model: ${MODEL}"

BPF_PATH=/home/tiger/byteprofile-analysis/analyze.py
PYTHON_FILE=/home/tiger/horovod_examples/tensorflow/tensorflow_synthetic_benchmark.py
TRACE_PATH=${BYTEPS_TRACE_DIR}/bps_trace_final.json
BPF_CMD="python3 ${BPF_PATH} --pretty --option collect --nccl_algo RING --path ${BYTEPS_TRACE_DIR} --platform TENSORFLOW --force"

### Start to train
if [ ! -d "${BYTEPS_TRACE_DIR}/host0" ]; then
	mkdir -p "${BYTEPS_TRACE_DIR}/host0"
else
	rm -rf ${BYTEPS_TRACE_DIR}/host0/*
fi
echo "Traces are stored at ${BYTEPS_TRACE_DIR}"

function funcReset {
	rm $TRACE_PATH
	rm -rf $BYTEPS_TRACE_DIR/host0/*
}

FIRST_RUN=1
function funcRunAndTest {
	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 \
		nohup python3 ${PYTHON_FILE} $@	
	echo "byteprofiler fp32: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	if [ ${FIRST_RUN} == "1" ]; then
		echo "Run the command: BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 python3 ${PYTHON_FILE} $@"
		${BPF_CMD} --sub_option amp_data_clct,save_names=fp32,model=resnet,platform=tf
		mv $BYTEPS_TRACE_DIR/host0/0/metadata.json $BYTEPS_TRACE_DIR/
	else
		${BPF_CMD} --sub_option amp_data_clct,save_names=None,model=resnet,platform=tf
	fi

	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 python3 ${PYTHON_FILE} --amp $@	
	echo "byteprofiler fp16: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	if [ ${FIRST_RUN} == "1" ]; then
		${BPF_CMD} --sub_option amp_data_clct,save_names=fp16,model=resnet,platform=tf
		FIRST_RUN=0
	else
		${BPF_CMD} --sub_option amp_data_clct,save_names=None,model=resnet,platform=tf
	fi
}

### Run with different batch size
bs_to_try=(4 8 16 32 64 128 256 512 1024 2048)
for(( id=0; id < "${#bs_to_try[@]}"; id++ ))
do
	batch_size=${bs_to_try[$id]}
	if [ "$PLATFORM" == "tf" ] || [ "$PLATFORM" == "TF" ] ; then
		ARGUMENTS="--batch-size ${batch_size}"
	elif [ "$PLATFORM" == "mx" ]; then
		ARGUMENTS="--batch-size ${batch_size} --log-interval 10 --model ${MODEL}"
	else
		exit
	fi 
	funcRunAndTest ${ARGUMENTS}
done




