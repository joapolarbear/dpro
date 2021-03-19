#!/bin/bash

### MXNet env
MXNET_CUDNN_AUTOTUNE_DEFAULT=0
MXNET_GPU_WORKER_NTHREADS=1
MXNET_EXEC_BULK_EXEC_TRAIN=0

##############################################################################
### Configuration

# MODEL="ResNet50"
MODEL="BertBase"
# MODEL="InceptionV3"
# MODEL="VGG16"
# MODEL="Bert256"

PLATFORM='TF'
echo "Platform: ${PLATFORM}, Model: ${MODEL}"

BPF_PATH=/home/tiger/byteprofile-analysis/analyze.py
TRACE_PATH=${BYTEPS_TRACE_DIR}/bps_trace_final.json
BPF_CMD="python3 ${BPF_PATH} --pretty --option collect --nccl_algo RING --path ${BYTEPS_TRACE_DIR} --platform TENSORFLOW --force"

function bert_env {
    export BPF_BATCH_PER_GPU="${BS:-32}"          
    export BPF_NUMSTEPS="${BPF_NUMSTEPS:-100}"
    export BERT_ZIP_DIR=/opt/tiger/bert/data/BERT-Base_uncase
    export BERT_BASE_DIR=$BERT_ZIP_DIR/uncased_L-12_H-768_A-12
    export MAX_SEQ_LENGTH=128
    export MAX_PREDICTIONS_PER_SEQ=20
}

function funcConfigBaseCMD {
	if [ "$MODEL" = "ResNet50" ] || [ "$MODEL" = "VGG16" ] || [ "$MODEL" = "InceptionV3" ]; then
		FILE_PATH="/home/tiger/horovod_examples/tensorflow/tensorflow_synthetic_benchmark.py"
	else
		FILE_PATH="/home/tiger/bert/run_pretraining.py"
	fi

	if [ "$MODEL" = "ResNet50" ]; then
		BASE_CMD="python3 ${FILE_PATH} --batch-size ${batch_size} --classes 1000"
	elif [ "$MODEL" = "VGG16" ]; then
		BASE_CMD="python3 ${FILE_PATH} --batch-size ${batch_size} --classes 1000 --model VGG16"
	elif [ "$MODEL" = "InceptionV3" ]; then
		BASE_CMD="python3 ${FILE_PATH} --batch-size ${batch_size} --classes 1000 --model InceptionV3"
	elif [ "$MODEL" = "Bert256" ]; then
		bert_env
		BASE_CMD="python3 ${FILE_PATH}  --train_batch_size=${batch_size}  --input_file=$BERT_BASE_DIR/tf_examples.tfrecord  --output_dir=$BERT_BASE_DIR/pretraining_output     --do_train=True     --do_eval=False     --bert_config_file=$BERT_BASE_DIR/bert_config.json  --max_seq_length=$MAX_SEQ_LENGTH     --max_predictions_per_seq=$MAX_PREDICTIONS_PER_SEQ     --num_train_steps=$BPF_NUMSTEPS    --num_warmup_steps=10     --learning_rate=2e-5 --synthetic --model bert_default"
	elif [ "$MODEL" = "BertBase" ]; then
		bert_env
		BASE_CMD="python3 ${FILE_PATH}  --train_batch_size=${batch_size}  --input_file=$BERT_BASE_DIR/tf_examples.tfrecord  --output_dir=$BERT_BASE_DIR/pretraining_output     --do_train=True     --do_eval=False     --bert_config_file=$BERT_BASE_DIR/bert_config.json  --max_seq_length=$MAX_SEQ_LENGTH     --max_predictions_per_seq=$MAX_PREDICTIONS_PER_SEQ     --num_train_steps=$BPF_NUMSTEPS    --num_warmup_steps=10     --learning_rate=2e-5 --synthetic --model bert_base"
	else
		echo "Invalid model: $MODEL"
		exit
	fi
}

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
	funcConfigBaseCMD
}

FIRST_RUN=1
function funcRunAndTest {
	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 \
		nohup ${BASE_CMD} $@	
	echo "dPRO fp32: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	echo "dPRO fp32: $@"
	echo "Run the command: ${BASE_CMD} $@"
	if [ ${FIRST_RUN} == "1" ]; then
		echo "${BPF_CMD} --sub_option amp_data_clct,save_names=fp32,model=resnet,platform=tf,showall=True"
		nohup ${BPF_CMD} --sub_option amp_data_clct,save_names=fp32,model=resnet,platform=tf,showall=True
		mv $BYTEPS_TRACE_DIR/host0/0 $BYTEPS_TRACE_DIR/.metadata/
		nvidia-smi >> $BYTEPS_TRACE_DIR/.metadata/config.txt
		echo "$bs_to_try" >> $BYTEPS_TRACE_DIR/.metadata/config.txt
	else
		nohup ${BPF_CMD} --sub_option amp_data_clct,save_names=None,model=resnet,platform=tf,showall=True
	fi

	funcReset
	BYTEPS_TRACE_DIR=$BYTEPS_TRACE_DIR/host0 BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=60 \
		nohup ${BASE_CMD} --amp $@	
	echo "dPRO fp16: $@" >> ${BYTEPS_TRACE_DIR}/avg.txt
	echo "dPRO fp16: $@"
	echo "Run the command: ${BASE_CMD} --amp$@"
	if [ ${FIRST_RUN} == "1" ]; then
		nohup ${BPF_CMD} --sub_option amp_data_clct,save_names=fp16,model=resnet,platform=tf,showall=True
		FIRST_RUN=0
	else
		nohup ${BPF_CMD} --sub_option amp_data_clct,save_names=None,model=resnet,platform=tf,showall=True
	fi
}

### Run with different batch size
bs_to_try=(4 8 16 32 64 128 256 512 1024 2048)
for(( id=0; id < "${#bs_to_try[@]}"; id++ ))
do
	batch_size=${bs_to_try[$id]}
	funcRunAndTest
done



