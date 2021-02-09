#!/bin/bash

### MXNet env
MXNET_CUDNN_AUTOTUNE_DEFAULT=0
MXNET_GPU_WORKER_NTHREADS=1
MXNET_EXEC_BULK_EXEC_TRAIN=0

### Profiling env
# BYTEPS_TRACE_ON=1 
# BYTEPS_TRACE_DIR=/opt/tiger/traces

MODEL='vgg11' 

BPF_PATH=/opt/tiger/byteprofile-analysis/analyze.py
EXMP_PATH=/opt/tiger/horovod_examples
PYTHON_FILE=${EXMP_PATH}/tensorflow_synthetic_benchmark.py
TRACE_PATH=${BYTEPS_TRACE_DIR}/bps_trace_final.json
BPF_CMD="python3 ${BPF_PATH} --pretty --option collect --nccl_algo RING --path ${BYTEPS_TRACE_DIR} --platform TENSORFLOW --force"

### Start to train
if [ ! -d "${BYTEPS_TRACE_DIR}/host0" ]; then
	mkdir -p "${BYTEPS_TRACE_DIR}/host0"
else
	rm -rf ${BYTEPS_TRACE_DIR}/host0/*
fi

function funcReset {
	rm $TRACE_PATH
	rm -rf $BYTEPS_TRACE_DIR/host0/*
}

function funcRunAndTestFirst {
	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 python3 ${PYTHON_FILE} $@	
	echo "byteprofiler fp32: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	${BPF_CMD} --sub_option amp_data_clct,save_names=fp32,model=resnet,platform=tf
	mv $BYTEPS_TRACE_DIR/host0/0/metadata.json $BYTEPS_TRACE_DIR/

	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 python3 ${PYTHON_FILE} --amp $@	
	echo "byteprofiler fp16: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	${BPF_CMD} --sub_option amp_data_clct,save_names=fp16,model=resnet,platform=tf
}

function funcRunAndTest {
	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 python3 ${PYTHON_FILE} $@	
	echo "byteprofiler fp32: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	${BPF_CMD} --sub_option amp_data_clct,save_names=None,model=resnet,platform=tf

	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 python3 ${PYTHON_FILE} --amp $@	
	echo "byteprofiler fp16: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	${BPF_CMD} --sub_option amp_data_clct,save_names=None,model=resnet,platform=tf
}


### Run with different batch size
START_BATCH_SIZE=4
for(( id=$START_BATCH_SIZE; id <= 128; id*=2 ))
do
	if [ "$1" == "tf" ]; then
		# CMD="--batch_size $id --use_synthetic_data --num_gpus=1 --max_train_steps=200 --train_epochs=1"
		CMD="--batch-size $id"
	elif [ "$1" == "mx" ]; then
		# CMD="--batch-size $id --log-interval 10 --model resnet50_v1"
		CMD="--batch-size $id --log-interval 10 --model ${MODEL}"
	else
		exit
	fi 
	if [ ${id} = $START_BATCH_SIZE ]; then
		funcRunAndTestFirst ${CMD}
	else
		funcRunAndTest ${CMD}
	fi
done

for(( id=256; id <= 1024; id+=128 ))
do
	if [ "$1" == "tf" ]; then
		# CMD="--batch_size $id --use_synthetic_data --num_gpus=1 --max_train_steps=200 --train_epochs=1"
		CMD="--batch-size $id"
	elif [ "$1" == "mx" ]; then
		# CMD="--batch-size $id --log-interval 10 --model resnet50_v1"
		CMD="--batch-size $id --log-interval 10 --model ${MODEL}"
	else
		exit
	fi 
	funcRunAndTest ${CMD}
done



