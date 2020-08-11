from platform.tensorflow.metadata import MetaInfo

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


class AMPPredictor:
	def __init__(self, meta_path):
		self.meta_info = MetaInfo(meta_path)

	def pred_amp_avg(self, op_name, _avg=None):
		op_type, S_mul, S_add, S_in, S_out, S_wei = self.meta_info.ret_tf_metadata(op_name)
		popt = OP_TYPES[op_type]["popt"]
		avg_fp32 = func_pred_time([FP32_FLOPS, S_mul, S_add, S_in, S_out, S_wei], *popt)
		avg_fp16 = func_pred_time([FP16_FLOPS, S_mul, S_add, S_in, S_out, S_wei], *popt)
		if _avg is not None:
			return _avg * avg_fp16 / avg_fp32
		else:
			return avg_fp16

	def pre_cast_time(self, op_name):
		op_type, S_mul, S_add, S_in, S_out, S_wei = self.meta_info.ret_tf_metadata(op_name)
		popt = OP_TYPES["Cast"]["popt"]
		in_cast = func_pred_time([0, 0, 0, S_in, S_in, 0], *popt)
		out_cast = func_pred_time([0, 0, 0, S_out, S_out, 0], *popt)
		wei_cast = func_pred_time([0, 0, 0, S_wei, S_wei, 0], *popt)
		return in_cast, out_cast, wei_cast

	def quantize(self, dag, op_name):
		in_cast, out_cast, wei_cast = self.pre_cast_time(op_name)
		assert dag.nodes[op_name].get("dtype", "fp32") == "fp32", op_name

		### handle parent nodes
		for u, v in dag.in_edges(op_name):
			if "AMPCastToFp32" in u:
				### not the boundary of mixed precision, remove the cast
				prevs = dag.in_edges(u)
				assert len(prevs) == 1, prevs
				dag.add_edge(prevs[0][0], op_name)
				dag.remove_edge(*prevs[0])
			else:
				### the boundary of mixed precision, add a cast op
				dag.add_edge(u, "%s.AMPCastToFp16"%op_name)
				dag.add_edge("%s.AMPCastToFp16"%op_name, op_name)
				# TODO (huhanpeng): need verify
				dag.nodes["%s.AMPCastToFp16"%op_name]["avg"] = wei_cast if "weight" in u.lower else in_cast

		### handle successors
		for succ in dag.successors(op_name):
			if "AMPCastToFp16" in u:
				nnexts = dag.successors(succ)
				assert len(nnexts) == 1, nnexts
				dag.add_edge(op_name, nnexts[0])
				self.remove_edge(succ, nnexts[0])
			else:
				dag.add_edge(op_name, "%s.AMPCastToFp32"%op_name)
				dag.add_edge("%s.AMPCastToFp32"%op_name, succ)
				# TODO (huhanpeng): need verify
				dag.nodes["%s.AMPCastToFp32"%op_name]["avg"] = out_cast

		### update the meta info of current node
		dag.nodes[op_name]["avg"] = self.pred_amp_avg(op_name, _avg=dag.nodes[op_name]["avg"])
		dag.nodes[op_name]["dtype"] = "fp16"

	def is_white_for_amp(self, dag, op_name):
		''' check whether an OP is finally white or not, according the propogation rules in AMP of TensorFlow '''
		if dag.nodes[op_name].get("is_white", "none") == "none":
			amp_color = self.meta_info.check_amp_lists(op_name)
			if amp_color == "white":
				### cache the intermediate result
				dag.nodes[op_name]["is_white"] = True
				return True
			if amp_color == "black":
				### cache the intermediate result
				dag.nodes[op_name]["is_white"] = False
				return False

			### need to further check the parent nodes
			is_white = True
			for u, _ in dag.in_edges(op_name):
				is_white &= self.is_white_for_amp(dag, u)
				if not is_white:
					break
			### cache the intermediate result
			dag.nodes[op_name]["is_white"] = is_white
			return is_white
		else:
			### return the cached results
			return dag.nodes[op_name]["is_white"]

	def is_need_amp(self, dag, op_name):
		''' check whether an OP need be quantized, only those with fp32 and in the final white list need be quantized'''
		if dag.nodes[op_name].get("dtype", "fp32") != "fp32":
			return False

		return self.is_white_for_amp(dag, op_name)

