import json
import os, sys
import numpy as np

''' The arguments for the function to calculate some info for each OP
	OP_ARGS = [bathc_size, C_in, C_out, width, kernel_size]
'''

OP_TYPES = {
			"Conv2D": {
				"popt": []
				},
			"MatMul": {
				"popt": []
			},
			"Cast": {
				"popt": []
			}
          }

FP32_FLOPS = 1
FP16_FLOPS = 2

def func_pred_time(xs, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3):
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

with open(tf_graph_path, 'r') as fp:
	tf_graph = json.load(fp)


def ret_tf_meta_info(op_name, tf_graph):
	'''
	Return:
		S_mul, S_add, S_in, S_out, S_weight
	'''
	inputs = tf_graph[op_name]["input"]
	outputs = tf_graph[op_name]["output"]
	op_type = tf_graph[op_name]["op"]
	if op_type == "Conv2D":
		assert len(outputs) == 1
		shape_ = outputs[0]["shape"]
		assert len(shape_) == 4
		N = shape_[0]
		P = shape_[1]
		Q = shape_[2]
		K = shape_[3]

		assert len(inputs) == 2
		C = None 		# input channel
		H = W = None 	# Input height/weight
		R = S = None 	# kernel size
		for input_ in inputs:
			shape_ = input_["shape"]
			assert len(shape_) == 4
			if "kernel" in input_["name"]:
				### weight
				R, S = shape_[0], shape_[1]
				if C is None:
					C = shape_[2]
				else:
					assert C == shape_[2]
				assert K == shape_[3]
			else:
				### Input
				assert shape_[0] == N
				H, W = shape_[1], shape_[2]
				if C is None:
					C = shape_[3]
				else:
					assert C == shape_[3]
		return op_type, N*K*P*Q*C*R*S, N*K*P*Q*(C*R*S-1), N*H*W*C, N*P*Q*K, R*S*C*K
	elif op_type == "MatMul":
		assert len(outputs) == 1
		shape_ = outputs[0]["shape"]
		assert len(shape_) == 2
		B = shape_[0]
		C_in = shape_[1]

		assert len(inputs) == 2
		C_out = None
		for input_ in inputs:
			shape_ = input_["shape"]
			assert len(shape_) == 2
			if "kernel" in input_["name"]:
				### weight
				assert C_in == shape_[0]
				if C_out is None:
					C_out = shape_[1]
				else:
					assert C_out == shape_[1]
			else:
				### Input
				assert shape_[0] == B
				if C_out is None:
					C_out = shape_[1]
				else:
					assert C_out == shape_[1]
		return op_type, B*C_in*C_out, B*(C_in-1)*C_out, B*C_in, B*C_out, C_in*C_out
	elif op_type == "Cast":
		assert len(outputs) == 1
		assert len(inputs) == 1
		shape_ = outputs[0]["shape"]

		return op_type, 0, 0, np.prod(inputs[0]["shape"]), np.prod(outputs[0]["shape"]), 0
	else:
		raise NotImplementedError("{} is not implemented yet.".format(op_name))


def pred_amp_avg(op_name, _avg=None):
	op_type, S_mul, S_add, S_in, S_out, S_wei = ret_tf_meta_info(op_name, tf_graph)
	popt = OP_TYPES[op_type]["popt"]
	avg_fp32 = func_pred_time([FP32_FLOPS, S_mul, S_add, S_in, S_out, S_wei], *popt)
	avg_fp16 = func_pred_time([FP16_FLOPS, S_mul, S_add, S_in, S_out, S_wei], *popt)
	if _avg is not None:
		return _avg * avg_fp16 / avg_fp32
	else:
		return avg_fp16

pred_amp_avg('network/resblock0_1/conv_0/conv2d/Conv2D')


