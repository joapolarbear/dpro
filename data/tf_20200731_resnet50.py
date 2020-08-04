"""
Platform: On one V100 GPU, single machine
Framework: Tensorflow 1.14, CUDA 10.2
Model/Dataset: 2conv + 2 dense with MNIST
	# ---------
	# Variables: name (type shape) [size]
	# ---------
	# conv_layer1/conv2d/kernel:0 (float32_ref 5x5x1x32) [800, bytes: 3200]
	# conv_layer1/conv2d/bias:0 (float32_ref 32) [32, bytes: 128]
	# conv_layer2/conv2d/kernel:0 (float32_ref 5x5x32x64) [51200, bytes: 204800]
	# conv_layer2/conv2d/bias:0 (float32_ref 64) [64, bytes: 256]
	# dense/kernel:0 (float32_ref 3136x1024) [3211264, bytes: 12845056]
	# dense/bias:0 (float32_ref 1024) [1024, bytes: 4096]
	# dense_1/kernel:0 (float32_ref 1024x10) [10240, bytes: 40960]
	# dense_1/bias:0 (float32_ref 10) [10, bytes: 40]
	# Total size of variables: 3274634
	# Total bytes of variables: 13098536
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit
import os, sys

NAMELIST_32 = None
NAMELIST_16 = None
DATA_32 = {}
DATA_16 = {}
DEFAULT_BATCH_SIZE_STR="B=256"
DEFAULT_BATCH_SIZE=int(DEFAULT_BATCH_SIZE_STR.split("=")[1])
DEFAULT_KENREL_SIZE=3

BATCH_LIST_VALUE = []

RST_DIR="./data_20200804_resnet50/v100"

def str2list(_list, dtype=str):
	assert dtype in [int, float, str]
	elems = _list.split("[")[1].split("]")[0].split(", ")

	if dtype == str:
		return [str(e.split("'")[1]) for e in elems]
	else:
		return [dtype(e) for e in elems]

with open(os.path.join(RST_DIR, "name.txt"), 'r') as fp:
	lines = fp.read().split("\n")
	if lines[-1] == "":
		lines = lines[:-1]
	for line in lines:
		if "fp32" in line:
			NAMELIST_32 = str2list(line.split(":")[1])
		elif "fp16" in line:
			NAMELIST_16 = str2list(line.split(":")[1])
		else:
			raise

with open(os.path.join(RST_DIR, "avg.txt"), 'r') as fp:
	lines = fp.read().split("\n")
	idx = 0
	while True:
		if "huhanpeng" in lines[idx]:
			batchsize = int(lines[idx].split("--batch_size")[1])
			if "fp32" in lines[idx]:
				_DATA = DATA_32
			elif "fp16" in lines[idx]:
				_DATA = DATA_16
			else:
				raise
			_DATA["B=%d"%batchsize] = str2list(lines[idx+1], dtype=float)
			BATCH_LIST_VALUE.append(batchsize)
			idx += 2
		else:
			idx += 1
		if idx >= len(lines):
			break
assert "network/resblock_3_1/conv_1/conv2d/Conv2D" in NAMELIST_32, "%s"%str(NAMELIST_32)

BATCH_LIST_VALUE = [e for e in BATCH_LIST_VALUE if e >=4]
BATCH_LIST_VALUE = sorted(BATCH_LIST_VALUE)
BATCH_LIST_STR = ["B=%d"%e for e in BATCH_LIST_VALUE]

def vary_batch_size(index, fp16=False):
	avg = []
	_DATA = DATA_16 if fp16 else DATA_32
	for e in BATCH_LIST_STR:
		avg.append(_DATA[e][index])
	return avg

def numOfMulInDense(B, C_in, C_out):
	return B * C_in * C_out

def numOfMulInConv(B, width, K, C_in, C_out):
	return B * width * width * K * K * C_in * C_out

def numOfAddInDense(B, C_in, C_out):
	return B * (C_in - 1) * C_out

def numOfAddInConv(B, width, K, C_in, C_out):
	return B * width * width * (K * K * C_in - 1) * C_out

def sizeInOfDense(B, C_in, C_out):
	return B * C_in

def sizeInOfConv(B, width, K, C_in, C_out):
	return B * width * width * C_in

def sizeOutOfDense(B, C_in, C_out):
	return B * C_out

def sizeOutOfConv(B, width, K, C_in, C_out):
	return B * width * width * C_out

def sizeWeiOfDense(B, C_in, C_out):
	return C_in * C_out

def sizeWeiOfConv(B, width, K, C_in, C_out):
	return K * K * C_in * C_out

def infoOfConv(B, width, K, C_in, C_out):
	return numOfMulInConv(B, width, K, C_in, C_out), \
		numOfAddInConv(B, width, K, C_in, C_out), \
		sizeInOfConv(B, width, K, C_in, C_out), \
		sizeOutOfConv(B, width, K, C_in, C_out), \
		sizeWeiOfConv(B, width, K, C_in, C_out)

def infoOfDense(B, C_in, C_out):
	return numOfMulInDense(B, C_in, C_out), \
		numOfAddInDense(B, C_in, C_out), \
		sizeInOfDense(B, C_in, C_out), \
		sizeOutOfDense(B, C_in, C_out), \
		sizeWeiOfDense(B, C_in, C_out)

def show_model_complexity():
	computeComplexity = [["conv_layer1", "conv_layer2", "dense", "dense1"], 
			[numOfMulInConv(DEFAULT_BATCH_SIZE, 28, DEFAULT_KENREL_SIZE, 1, 32),
				numOfMulInConv(DEFAULT_BATCH_SIZE, 14, DEFAULT_KENREL_SIZE, 32, 64),
				numOfMulInDense(DEFAULT_BATCH_SIZE, 7*7*64, DEFAULT_DENSE_SIZE),
				numOfMulInDense(DEFAULT_BATCH_SIZE, DEFAULT_DENSE_SIZE, 10)],
			[numOfAddInConv(DEFAULT_BATCH_SIZE, 28, DEFAULT_KENREL_SIZE, 1, 32),
				numOfAddInConv(DEFAULT_BATCH_SIZE, 14, DEFAULT_KENREL_SIZE, 32, 64),
				numOfAddInDense(DEFAULT_BATCH_SIZE, 7*7*64, DEFAULT_DENSE_SIZE),
				numOfAddInDense(DEFAULT_BATCH_SIZE, DEFAULT_DENSE_SIZE, 10)]]
	for i in range(len(computeComplexity[0])):
		print("%-20s mul:%-20d add: %-20d"%(computeComplexity[0][i], computeComplexity[1][i], computeComplexity[2][i]))

def init_fig_base(cnt):
	w = math.ceil(math.sqrt(cnt))
	h = math.ceil(cnt / w)
	fig_base = w * 100 + h * 10 + 1
	return fig_base, 0

# show_model_complexity()

OP_NAMES = [
				'network/resblock0_1/conv_0/conv2d/Conv2D',
				'network/resblock1_1/conv_0/conv2d/Conv2D',
				'network/resblock2_1/conv_0/conv2d/Conv2D', 
				'network/resblock_3_1/conv_1/conv2d/Conv2D', 			
			]
OP_LABELS = ["/".join(n.split("/")[1:]) for n in OP_NAMES]

model_size = np.array([
		list(zip(*[infoOfConv(b, 32, DEFAULT_KENREL_SIZE, 32, 32) for b in BATCH_LIST_VALUE])),
		list(zip(*[infoOfConv(b, 16, DEFAULT_KENREL_SIZE, 64, 64) for b in BATCH_LIST_VALUE])),
		list(zip(*[infoOfConv(b, 8, DEFAULT_KENREL_SIZE, 128, 128) for b in BATCH_LIST_VALUE])),
		list(zip(*[infoOfConv(b, 4, DEFAULT_KENREL_SIZE, 256, 256) for b in BATCH_LIST_VALUE])),
	])
intensity = model_size[:, 0, :] / (model_size[:, 2, :] + model_size[:, 3, :] + model_size[:, 4, :])

THRESHOLD = 0.00 # in ms

def plot_varyB_resut(S_mul=False, S_add=False, S_in=False, S_out=False, S_wei=False):
	plt.figure(num=1, figsize=(8, 6))

	x_axis_idx = None
	if S_mul:
		x_axis_idx = 0
	elif S_add:
		x_axis_idx = 1
	elif S_in:
		x_axis_idx = 2
	elif S_out:
		x_axis_idx = 3
	elif S_wei:
		x_axis_idx = 4

	x_axis_names = ["S_mul", "S_add", "S_in", "S_out", "S_weight"]
	x_axis_name = "Batch Size (B)" if x_axis_idx is None else x_axis_names[x_axis_idx]

	fig_base, _ = init_fig_base(len(OP_NAMES))

	def __plot(fig_base, idx):
		ax = plt.subplot(fig_base + idx)
		x_axis = BATCH_LIST_VALUE if x_axis_idx is None else model_size[idx][x_axis_idx]
		avgs = vary_batch_size(NAMELIST_32.index(OP_NAMES[idx]))
		ax.plot(x_axis, avgs, marker='.', label=OP_LABELS[idx] + "_fp32")
		avgs_16 = vary_batch_size(NAMELIST_16.index(OP_NAMES[idx]), fp16=True)
		ax.plot(x_axis, avgs_16, marker='^', label=OP_LABELS[idx] + "_fp16")

		plt.legend()
		plt.ylabel('Average Time (ms)')
		plt.xlabel(x_axis_name)

	for i in range(len(OP_NAMES)):
		__plot(fig_base, i)

	plt.legend()
	plt.ylabel('Average Time (ms)')
	plt.xlabel(x_axis_name)

	plt.show()

def plot_varyB_intensity():
	plt.figure(num=1, figsize=(8, 6))

	# x_axis_names = ["S_mul", "S_add", "S_in", "S_out", "S_weight"]
	x_axis_name = "Batch Size (B)"

	def __plot(op_id):
		ax = plt.subplot(241 + 2 * op_id)
		avgs = vary_batch_size(NAMELIST_32.index(OP_NAMES[op_id]))
		avgs_16 = vary_batch_size(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)

		flops_32 = model_size[op_id, 0, :] / np.array(avgs)
		flops_16 = model_size[op_id, 0, :] / np.array(avgs_16)
		ax.plot(intensity[op_id], flops_32, '.-', label=OP_LABELS[op_id]+"_fp32_flops")
		ax.plot(intensity[op_id], flops_16, '.-', label=OP_LABELS[op_id]+"_fp16_flops")
		plt.legend()
		plt.ylabel('FLOPS')
		plt.xlabel("arithmetic intensity")

		ax = plt.subplot(242 + 2 * op_id)
		ax.plot(BATCH_LIST_VALUE, intensity[op_id], '.-', label=OP_LABELS[op_id]+"_intensity")
		plt.legend()
		plt.ylabel('Intensity')
		plt.xlabel(x_axis_name)
		op_id + 1

	for op_id in range(4):
		__plot(op_id)

	plt.show()

def plot_varyB_intensity_combine():
	plt.figure(num=1, figsize=(8, 6))

	# x_axis_names = ["S_mul", "S_add", "S_in", "S_out", "S_weight"]
	x_axis_name = "Batch Size (B)"

	def __plot(op_id):
		ax = plt.subplot(211 + op_id)

		avgs = vary_batch_size(NAMELIST_32.index(OP_NAMES[op_id]))
		avgs_16 = vary_batch_size(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)
		flops_32 = model_size[op_id, 0, :] / np.array(avgs)
		flops_16 = model_size[op_id, 0, :] / np.array(avgs_16)
		ax.plot(intensity[op_id], flops_32, '.-', label=OP_LABELS[op_id]+"_fp32_flops")
		ax.plot(intensity[op_id], flops_16, '.-', label=OP_LABELS[op_id]+"_fp16_flops")

		op_id += 1

		avgs = vary_batch_size(NAMELIST_32.index(OP_NAMES[op_id]))
		avgs_16 = vary_batch_size(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)
		flops_32 = model_size[op_id, 0, :] / np.array(avgs)
		flops_16 = model_size[op_id, 0, :] / np.array(avgs_16)
		ax.plot(intensity[op_id], flops_32, '.-', label=OP_LABELS[op_id]+"_fp32_flops")
		ax.plot(intensity[op_id], flops_16, '.-', label=OP_LABELS[op_id]+"_fp16_flops")

		plt.legend()
		plt.ylabel('FLOPS')
		plt.xlabel("arithmetic intensity")

	for op_id in range(2):
		__plot(op_id)

	plt.show()

def plot_avg_accum_distribution():
	plt.figure(num=1, figsize=(8, 6))
	def __plot(op_id):
		ax = plt.subplot(211 + op_id)

		avgs = vary_batch_size(NAMELIST_32.index(OP_NAMES[op_id]))
		avgs_16 = vary_batch_size(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)

		op_id += 1

		avgs_ = vary_batch_size(NAMELIST_32.index(OP_NAMES[op_id]))
		avgs_16_ = vary_batch_size(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)

		avgs = np.concatenate((avgs, avgs_))
		avgs_16 = np.concatenate((avgs_16, avgs_16_))

		def _gen(_array):
			assert len(_array.shape)==1
			_array = np.sort(_array)
			_len = _array.shape[0]
			dist = []
			ret = None
			for i in range(_len):
				dist.append(100 * i / _len)
				if ret is None and _array[i] > THRESHOLD:
					ret = 100 * i / _len
			return _array, np.array(dist), ret

		x1, y1, ret1 = _gen(avgs)
		x2, y2, ret2 = _gen(avgs_16)

		ax.plot(x1, y1, label="fp32" + ", fp32 == %f: %6.4f %%"%(THRESHOLD, ret1))
		ax.plot(x2, y2, label="fp16" + ", fp16 == %f: %6.4f %%"%(THRESHOLD, ret2))

		# ax.text(0.1, ret1, "fp32: {}, {} %".format(0.1, ret1))
		# ax.text(0.1, ret2, "fp16: {}, {} %".format(0.1, ret2))

		plt.legend()
		plt.ylabel('Cumulative Distribution (%)')
		plt.xlabel("Execution time (ms)")

	for op_id in range(2):
		__plot(op_id)

	plt.show()

def plot_varyK_result(S_mul=False, S_add=False, S_in=False, S_out=False, S_wei=False):
	plt.figure(num=1, figsize=(8, 6))

	cnt = 1 + int(S_mul) + int(S_add) + int(S_in) + int(S_out) + int(S_wei)
	fig_base, fig_idx = init_fig_base(cnt)
	
	model_size = [list(zip(*[infoOfConv(DEFAULT_BATCH_SIZE, 28, k, 1, 32) for k in KENREL_LIST_VALUE])),
		list(zip(*[infoOfConv(DEFAULT_BATCH_SIZE, 14, k, 32, 64) for k in KENREL_LIST_VALUE]))]

	def __plot(x_axis, x_axis2, x_name, _fig_idx):
		ax = plt.subplot(fig_base+_fig_idx)
		ax.plot(x_axis, vary_kernel_size(NAMELIST_32.index('conv_layer1/conv2d/Conv2D')), marker='.', label="conv1 (K*K*1*32)")
		ax.plot(x_axis, vary_kernel_size(NAMELIST_16.index('conv_layer1/conv2d/Conv2D'), fp16=True), marker='^', label="conv1 (K*K*1*32) + fp16")
		ax.plot(x_axis2[:6], vary_kernel_size(NAMELIST_32.index('conv_layer2/conv2d/Conv2D'))[:6], marker='.', label="conv2 (K*K*32*64)")
		ax.plot(x_axis2[:6], vary_kernel_size(NAMELIST_16.index('conv_layer2/conv2d/Conv2D'), fp16=True)[:6], marker='^', label="conv2 (K*K*32*64) + fp16")
		plt.legend()
		plt.ylabel('Average Time (ms)')
		plt.xlabel(x_name)
		return _fig_idx + 1

	fig_idx = __plot(KENREL_LIST_VALUE, KENREL_LIST_VALUE, "Kernel Size (K)", fig_idx)

	if S_mul:
		fig_idx = __plot(model_size[0][0], model_size[1][0], "S_mul", fig_idx)

	if S_add:
		fig_idx = __plot(model_size[0][1], model_size[1][1], "S_add", fig_idx)

	if S_in:
		fig_idx = __plot(model_size[0][2], model_size[1][2], "S_in", fig_idx)

	if S_out:
		fig_idx = __plot(model_size[0][3], model_size[1][3], "S_out", fig_idx)

	if S_wei:
		fig_idx = __plot(model_size[0][4], model_size[1][4], "S_wei", fig_idx)

	plt.show()

def plot_varyD_result():
	plt.figure(num=1, figsize=(8, 6))
	ax = plt.subplot(121)
	ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_32.index('dense/MatMul')), marker='.', label="dense (3136*D)")
	ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_16.index('dense/MatMul'), fp16=True), marker='^', label="dense (3136*D) + fp16")
	plt.legend()
	plt.ylabel('Average Time (ms)')
	plt.xlabel("Dense Size (D)")

	ax = plt.subplot(122)
	ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_32.index('dense_1/MatMul')), marker='.', label="dense1 (D*10)")
	ax.plot(DENSE_LIST_VALUE, vary_dense_size(NAMELIST_16.index('dense_1/MatMul'), fp16=True), marker='^', label="dense1 (D*10) + fp16")
	plt.legend()
	plt.ylabel('Average Time (ms)')
	plt.xlabel("Dense Size (D)")
	plt.show()

def plot_varyB_resut_of_cast():
	plt.figure(num=1, figsize=(8, 6))

	ax = plt.subplot(221)
	avgs_16 = vary_batch_size(NAMELIST_16.index('conv_layer1/Cast_1'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="conv_layer1/Cast_1:fp16->fp32")
	avgs_16 = vary_batch_size(NAMELIST_16.index('conv_layer1/conv2d/Conv2D'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="conv1 (5*5*1*32) + fp16")
	plt.legend()
	plt.ylabel('Average Time (ms)')
	plt.xlabel("Batch Size (B)")


	ax = plt.subplot(222)
	avgs_16 = vary_batch_size(NAMELIST_16.index('conv_layer2/Cast'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="conv_layer2/Cast:fp32->fp16")
	avgs_16 = vary_batch_size(NAMELIST_16.index('conv_layer2/conv2d/Conv2D'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="conv2 (5*5*32*64) + fp16")
	plt.legend()
	plt.ylabel('Average Time (ms)')
	plt.xlabel("Batch Size (B)")

	ax = plt.subplot(223)
	avgs_16 = vary_batch_size(NAMELIST_16.index('Cast_2'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="Cast_2(dense):fp16->fp32")
	avgs_16 = vary_batch_size(NAMELIST_16.index('dense/MatMul'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="dense (3136*1024) + fp16")
	plt.legend()
	plt.ylabel('Average Time (ms)')
	plt.xlabel("Batch Size (B)")

	ax = plt.subplot(224)
	avgs_16 = vary_batch_size(NAMELIST_16.index('Cast_3'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='.', label="Cast_3(dense1):fp32->fp16")
	avgs_16 = vary_batch_size(NAMELIST_16.index('dense_1/MatMul'), fp16=True)
	ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label="dense1 (1024*10) + fp16")
	plt.legend()
	plt.ylabel('Average Time (ms)')
	plt.xlabel("Batch Size (B)")

	plt.show()

# plot_varyK_result(S_mul=True, S_add=True, S_in=True, S_out=True, S_wei=True)
# plot_varyD_result()
# plot_varyB_resut(S_mul=False, S_add=False, S_in=False, S_out=False, S_wei=False)
# plot_varyB_intensity()
# plot_varyB_intensity_combine()
# plot_varyB_resut_of_cast()
# plot_avg_accum_distribution()
# raise


####################################################################################################
#############################        Start to Fit          #########################################
####################################################################################################

def exct_filter(target, others):
	assert len(target.shape) == 1
	_filter = np.where(target >= THRESHOLD)
	if len(others.shape) == 1:
		return target[_filter], others[_filter]
	else:
		return target[_filter], others[:, _filter].reshape(list(others.shape[:-1]) + [-1])

def predict_error(_list, _list_pred):
	_list_pred = np.array(_list_pred)
	_list = np.array(_list)
	_list, _list_pred = exct_filter(_list, _list_pred)

	if len(_list) == 0:
		return None, "Original time is too small. Ignore!!!"

	diff = np.abs(_list_pred - _list) / _list
	return diff, "%f %%"%(np.average(diff * 100))

def calculationSizeAndGFLOPS2time(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3):
	'''
	gflops:
		We only need a relative value of gflops, 
		i.e., if we know fp32's is twice of fp16's, we can just fp32's = 2 and fp16's = 1,
		the scale is hidden in the a2 
		x[0]: relative gflops
		x[1]: num of multiplication
		x[2]: num of addition
		x[3]: input size
		x[4]: output size
		x[5]: weight size
	'''
	gflops = xs[0]
	S_mul = xs[1]
	S_add = xs[2]
	wei_S_all = a4 * xs[3] + a5 * xs[4] + a6 * xs[5]
	wei_S_all2 = a7 * xs[3] + a8 * xs[4] + a9 * xs[5]
	return (a1 * S_mul + b1) / (a3 * gflops + b2) + wei_S_all / gflops + b3 + gflops * wei_S_all2 
	# return (a1 * xs[1] + b1) / (np.exp(a3 * xs[0] * (a4 * xs[3] + a5 * xs[4] + a6 * xs[5]) * (xs[1]/(xs[3] + xs[4] + a7 * xs[3] + a8 * xs[4] + a9 * xs[5]))) + b2) + b3 

def fit_test(xs, a1, a2, a3, a4, b1):
	gflops = xs[0]
	S_mul = xs[1]
	S_add = xs[2]
	intensity = xs[1] / (xs[3] + xs[4] + xs[5])
	return (a1 * S_mul + b1) * (1 / (a2 * gflops * intensity) + 1 / (a3 * gflops) + a4 * gflops / intensity)
		

def wrap_curve_fit(xs, ys):
	assert isinstance(xs, list) and isinstance(ys, list)
	return curve_fit(time2batch_size, [0] + xs, [0] + ys)

lower_bounds = tuple([0]*9 + [-np.inf]*3)
# lower_bounds = tuple([0]*4 + [-np.inf]*1)
up_bounds = tuple(len(lower_bounds) * [np.inf])
p0=[0]*len(lower_bounds)
FIT_FUNC = calculationSizeAndGFLOPS2time
# FIT_FUNC = fit_test

is_show = [True, True, True, True, False, False, False, False]
GFLOPS_FP32 = 1
GFLOPS_FP16 = 2

def fit_with_S_cal_gflops(is_show):
	avgsList = []
	gflopsList = []
	S_mul_list = []
	S_add_list = []
	S_in_list = []
	S_out_list = []
	S_wei_list = []

	def __record_xdata(S_mul, S_add, S_in, S_out, S_wei, gflops):
		S_mul_list.append(S_mul)
		S_add_list.append(S_add)
		S_in_list.append(S_in)
		S_out_list.append(S_out)
		S_wei_list.append(S_wei)
		gflopsList.append(gflops)

	w_ = 32
	cin = 32

	for i in range(len(OP_NAMES)):
		if is_show[i]:
			avgsList += vary_batch_size(NAMELIST_32.index(OP_NAMES[i]))
			for b in BATCH_LIST_VALUE:
				S_mul, S_add, S_in, S_out, S_wei = infoOfConv(b, w_, DEFAULT_KENREL_SIZE, cin, cin)
				__record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP32)
			avgsList += vary_batch_size(NAMELIST_16.index(OP_NAMES[i]), fp16=True)
			for b in BATCH_LIST_VALUE:
				S_mul, S_add, S_in, S_out, S_wei = infoOfConv(b, w_, DEFAULT_KENREL_SIZE, cin, cin)
				__record_xdata(S_mul, S_add, S_in, S_out, S_wei, GFLOPS_FP16)
		w_ /= 2
		cin *=2

	xdata = np.array([gflopsList, S_mul_list, S_add_list, S_in_list, S_out_list, S_wei_list])
	ydata = np.array(avgsList)
	_ydata, _xdata = exct_filter(ydata, xdata)
	popt, pcov = curve_fit(FIT_FUNC, _xdata, _ydata, 
		bounds=(lower_bounds, up_bounds), p0=p0, maxfev=10000)
	return popt, pcov, xdata, ydata

popt, pcov, xdata, ydata = fit_with_S_cal_gflops(is_show)
print(popt)
UNIT_LEN = len(BATCH_LIST_VALUE)

def plot_intensity2flops():
	assert not is_show[4]
	intensity = xdata[1] / (xdata[3] + xdata[4] + xdata[5])
	flops = xdata[1] / ydata

	from mpl_toolkits.mplot3d import Axes3D
	plt.figure(num=1, figsize=(8, 6))
	fig = plt.figure()
	ax = fig.add_subplot(121, projection='3d')
	ax.scatter(intensity, xdata[0], flops)
	plt.xlabel("arithmetic intensity")
	plt.ylabel("precision")

	ax = plt.subplot(122)
	index32 = np.where(xdata[0,:]==GFLOPS_FP32)
	index16 = np.where(xdata[0,:]==GFLOPS_FP16)
	intensity32 = intensity[index32]
	flops_32_to_16 = flops[index32] / flops[index16]
	ax.plot(intensity32, flops_32_to_16)
	plt.xlabel("arithmetic intensity")
	plt.ylabel("flops 32 / flops 16")
	plt.show()

def plot_3d_fit_result():
	from mpl_toolkits.mplot3d import Axes3D
	xdata = np.array([gflopsList, S_mul_list, S_in_list, S_out_list, S_wei_list])
	pred = FIT_FUNC(xdata, *popt)
	plt.figure(num=1, figsize=(8, 6))
	fig = plt.figure()
	fig_idx = 0

	ax = fig.add_subplot(221, projection='3d')
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
	plt.legend()
	plt.xlabel("Sin Conv1 32")
	plt.ylabel("Swei")
	fig_idx += 1

	ax = fig.add_subplot(222, projection='3d')
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
	plt.legend()
	plt.xlabel("Sin Conv1 16")
	plt.ylabel("Swei")
	fig_idx += 1

	ax = fig.add_subplot(223, projection='3d')
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
	plt.legend()
	plt.xlabel("Sin Conv2 32")
	plt.ylabel("Swei")
	fig_idx += 1

	ax = fig.add_subplot(224, projection='3d')
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		avgsList[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="real")
	ax.scatter(S_in_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		S_wei_list[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], 
		pred[fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], label="predict")
	plt.legend()
	plt.xlabel("Sin Conv2 16")
	plt.ylabel("Swei")
	fig_idx += 1

	plt.show()

def plot_2d_fit_result(is_show):
	plt.figure(num=1, figsize=(8, 6))

	ratio_sum = []
	cnt = sum([int(i) for i in is_show])
	fig_base, fig_idx = init_fig_base(cnt)

	def __plot(op_id, fig_base, fig_idx, only_16=False):
		ax = plt.subplot(fig_base)
		if not only_16:
			avgs = vary_batch_size(NAMELIST_32.index(OP_NAMES[op_id]))
			ax.plot(BATCH_LIST_VALUE, avgs, marker='.', label=OP_LABELS[op_id]+"_fp32")
			avgs_pred = FIT_FUNC(xdata[:, fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], *popt)
			ax.plot(BATCH_LIST_VALUE, avgs_pred, "--", label=OP_LABELS[op_id]+"_fp32"+"_pred")
			diff, ratio = predict_error(avgs, avgs_pred)
			print(OP_LABELS[op_id]+"_fp32", ratio, diff)
			fig_idx += 1
			if "%" in ratio:
				ratio_sum.append(float(ratio.split("%")[0]))

		avgs_16 = vary_batch_size(NAMELIST_16.index(OP_NAMES[op_id]), fp16=True)
		ax.plot(BATCH_LIST_VALUE, avgs_16, marker='^', label=OP_LABELS[op_id]+"_fp16")
		avgs_pred = FIT_FUNC(xdata[:, fig_idx*UNIT_LEN:(fig_idx+1)*UNIT_LEN], *popt)
		ax.plot(BATCH_LIST_VALUE, avgs_pred, "--", label=OP_LABELS[op_id]+"_fp16"+"_pred")
		diff, ratio = predict_error(avgs_16, avgs_pred)
		print(OP_LABELS[op_id]+"_fp16", ratio, diff)
		fig_idx += 1
		if "%" in ratio:
			ratio_sum.append(float(ratio.split("%")[0]))
		
		plt.legend()
		plt.ylabel('Average Time (ms)')
		plt.xlabel("Batch Size (B)")
		return fig_base+1, fig_idx

	for op_id in range(4):
		if is_show[op_id]:
			fig_base, fig_idx = __plot(op_id, fig_base, fig_idx)
	
	print("average error: %f %%"%(sum(ratio_sum)/len(ratio_sum)))
	plt.show()

# plot_intensity2flops()
plot_2d_fit_result(is_show)




