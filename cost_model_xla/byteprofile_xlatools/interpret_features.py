import os
import sys
import code

if len(sys.argv) != 2:
    print("Usage: interpret_features.py PATH_TO_FEATURE.TXT")
    exit(0)

op_codes = """xla::HloOpcode::kAbs, xla::HloOpcode::kAdd, 
        xla::HloOpcode::kAddDependency, xla::HloOpcode::kAfterAll, xla::HloOpcode::kAllReduce,
        xla::HloOpcode::kAllToAll, xla::HloOpcode::kAtan2, xla::HloOpcode::kBatchNormGrad, 
        xla::HloOpcode::kBatchNormInference, xla::HloOpcode::kBatchNormTraining, 
        xla::HloOpcode::kBitcast, xla::HloOpcode::kBitcastConvert, xla::HloOpcode::kBroadcast, 
        xla::HloOpcode::kCall, xla::HloOpcode::kCeil, xla::HloOpcode::kCholesky, 
        xla::HloOpcode::kClamp, xla::HloOpcode::kCollectivePermute, xla::HloOpcode::kClz, 
        xla::HloOpcode::kCompare, xla::HloOpcode::kComplex, xla::HloOpcode::kConcatenate,
        xla::HloOpcode::kConditional, xla::HloOpcode::kConstant, xla::HloOpcode::kConvert,
        xla::HloOpcode::kConvolution, xla::HloOpcode::kCopy, xla::HloOpcode::kCos, 
        xla::HloOpcode::kCustomCall, xla::HloOpcode::kDivide, xla::HloOpcode::kDomain,
        xla::HloOpcode::kDot, xla::HloOpcode::kDynamicSlice, xla::HloOpcode::kDynamicUpdateSlice,
        xla::HloOpcode::kExp, xla::HloOpcode::kExpm1, xla::HloOpcode::kFft, xla::HloOpcode::kFloor,
        xla::HloOpcode::kFusion, xla::HloOpcode::kGather, xla::HloOpcode::kGetDimensionSize, 
        xla::HloOpcode::kGetTupleElement, xla::HloOpcode::kImag, xla::HloOpcode::kInfeed, 
        xla::HloOpcode::kIota, xla::HloOpcode::kIsFinite, xla::HloOpcode::kLog, 
        xla::HloOpcode::kLog1p, xla::HloOpcode::kAnd, xla::HloOpcode::kNot, xla::HloOpcode::kOr,
        xla::HloOpcode::kXor, xla::HloOpcode::kMap, xla::HloOpcode::kMaximum, xla::HloOpcode::kMinimum,
        xla::HloOpcode::kMultiply, xla::HloOpcode::kNegate, xla::HloOpcode::kOutfeed, 
        xla::HloOpcode::kPad, xla::HloOpcode::kParameter, xla::HloOpcode::kPower, 
        xla::HloOpcode::kReal, xla::HloOpcode::kRecv, xla::HloOpcode::kRecvDone, 
        xla::HloOpcode::kReduce, xla::HloOpcode::kReducePrecision, xla::HloOpcode::kReduceWindow, 
        xla::HloOpcode::kRemainder, xla::HloOpcode::kReplicaId, xla::HloOpcode::kReshape,
        xla::HloOpcode::kReverse, xla::HloOpcode::kRng, xla::HloOpcode::kRoundNearestAfz,
        xla::HloOpcode::kRsqrt, xla::HloOpcode::kScatter, xla::HloOpcode::kSelect, 
        xla::HloOpcode::kSelectAndScatter, xla::HloOpcode::kSend, xla::HloOpcode::kSendDone,
        xla::HloOpcode::kShiftLeft, xla::HloOpcode::kShiftRightArithmetic, 
        xla::HloOpcode::kShiftRightLogical, xla::HloOpcode::kSign, xla::HloOpcode::kSin, 
        xla::HloOpcode::kSlice, xla::HloOpcode::kSort, xla::HloOpcode::kSqrt, xla::HloOpcode::kSubtract,
        xla::HloOpcode::kTanh, xla::HloOpcode::kTrace, xla::HloOpcode::kTranspose,
        xla::HloOpcode::kTriangularSolve, xla::HloOpcode::kTuple, xla::HloOpcode::kTupleSelect,
        xla::HloOpcode::kWhile"""

op_codes = [op_code.strip().split("::")[-1] for op_code in op_codes.split(",")]

feature_names = ["orig_seconds", "total_flops", "total_transcendental", "total_bytes", "total_opt_seconds"]

for op_code in op_codes:
    feature_names.append(op_code+":num_ops")
    feature_names.append(op_code+":avg_flops")
    feature_names.append(op_code+":total_flops")
    feature_names.append(op_code+":avg_transcendental")
    feature_names.append(op_code+":total_transcendental")
    feature_names.append(op_code+":avg_bytes")
    feature_names.append(op_code+":total_bytes")
    feature_names.append(op_code+":avg_opt_seconds")
    feature_names.append(op_code+":total_opt_seconds")
    if op_code == "kFusion":
        feature_names.append(op_code+"instr_count")

feature_vec = []
feature_dict = {}

with open(sys.argv[1], "r") as f:
    count = 0
    for line in f:
        feature_vec.append(float(line.strip()))
        feature_dict[feature_names[count]] = feature_vec[count]
        if feature_vec[count] != 0:
            print(feature_names[count], feature_vec[count])
        count += 1

code.interact(local=locals())



