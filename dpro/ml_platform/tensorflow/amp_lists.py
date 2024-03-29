
whitelist = [
        #if CUDA_VERSION >= 9010  // Fp16 BatchMatMul is slow before CUDA 9.1.
        "BatchMatMul",
        "BlockLSTM", "BlockLSTMGrad", "Conv2D", "Conv2DBackpropFilter",
        "Conv2DBackpropInput",

        # # TODO(benbarsdell): Enable these when Tensor Core kernels are
        # # available for 3D convolutions.
        # "Conv3D",
        # "Conv3DBackpropFilter",
        # "Conv3DBackpropFilterV2",
        # "Conv3DBackpropInput",
        # "Conv3DBackpropInputV2",
        # "CudnnRNN", "CudnnRNNBackprop", "CudnnRNNBackpropV2",
        # "CudnnRNNBackpropV3", "CudnnRNNV2", "CudnnRNNV3", "GRUBlockCell",
        # "GRUBlockCellGrad", "LSTMBlockCell", "LSTMBlockCellGrad",


        # # TODO(benbarsdell): Enable these when fast and safe fp16 kernels are
        # available for depthwise convolutions.
        # "DepthwiseConv2dNative",
        # "DepthwiseConv2dNativeBackpropFilter",
        # "DepthwiseConv2dNativeBackpropInput",

        "MatMul",
]

greylist = [
        "Add",
        "AddN",
        "AddV2",
        "AvgPool",
        "AvgPool3D",
        "AvgPool3DGrad",
        "AvgPoolGrad",
        "BiasAdd",
        "BiasAddGrad",
        "BiasAddV1",
        "Elu",
        "EluGrad",
        "Erf",
        "Erfc",
        "FloorDiv",
        "FusedBatchNormV2",
        "FusedBatchNormGradV2",
        "FusedBatchNormV3",
        "FusedBatchNormGradV3",
        "Inv",
        "LeakyRelu",
        "LeakyReluGrad",
        "Mul",
        "Prod",
        "RealDiv",
        "Reciprocal",
        "Sigmoid",
        "SigmoidGrad",
        "Softplus",
        "SoftplusGrad",
        "Sqrt",
        "Sub",
        "Tanh",
        "TanhGrad",
]

blacklist = [
        "Exp",
        "Expm1",
        "L2Loss",
        "Log",
        "Log1p",
        "LogSoftmax",
        "Mean",
        "Pow",
        "SaveV2",
        "Softmax",
        "SoftmaxCrossEntropyWithLogits",
        "SparseSoftmaxCrossEntropyWithLogits",
        "Sum",
]

clearlist = [
        "Abs",
        "ArgMax",
        "ArgMin",
        "BatchToSpace",
        "BatchToSpaceND",
        "BroadcastTo",
        "Ceil",
        "CheckNumerics",
        "ClipByValue",
        "Concat",
        "ConcatV2",
        "DepthToSpace",
        "DynamicPartition",
        "DynamicStitch",
        "Enter",
        "EnsureShape",
        "Equal",
        "Exit",
        "ExpandDims",
        "Fill",
        "Floor",
        "Gather",
        "GatherNd",
        "GatherV2",
        "Greater",
        "GreaterEqual",
        "Identity",
        "IdentityN",
        "IsFinite",
        "IsInf",
        "IsNan",
        "Less",
        "LessEqual",
        "Max",
        "MaxPool",
        "MaxPool3D",
        "MaxPool3DGrad",
        "MaxPool3DGradGrad",
        "MaxPoolGrad",
        "MaxPoolGradGrad",
        "MaxPoolGradGradV2",
        "MaxPoolGradV2",
        "MaxPoolV2",
        "Maximum",
        "Merge",
        "Min",
        "Minimum",
        "MirrorPad",
        "MirrorPadGrad",
        "Neg",
        "NextIteration",
        "NotEqual",
        "OneHot",
        "OnesLike",
        "Pack",
        "Pad",
        "PadV2",
        "PreventGradient",
        "Rank",
        "Relu",
        "Relu6",
        "Relu6Grad",
        "ReluGrad",
        "Reshape",
        "ResizeNearestNeighbor",
        "ResizeNearestNeighborGrad",
        "Reverse",
        "ReverseSequence",
        "ReverseV2",
        "Round",
        "Select",
        "Shape",
        "ShapeN",
        "Sign",
        "Size",
        "Slice",
        "Snapshot",
        "SpaceToBatch",
        "SpaceToBatchND",
        "SpaceToDepth",
        "Split",
        "SplitV",
        "Squeeze",
        "StackPopV2",
        "StackPushV2",
        "StopGradient",
        "StridedSlice",
        "StridedSliceGrad",
        "Switch",
        "TensorArrayConcatV3",
        "TensorArrayGatherV3",
        "TensorArrayReadV3",
        "TensorArrayScatterV3",
        "TensorArraySplitV3",
        "TensorArrayWriteV3",
        "Tile",
        "TopK",
        "TopKV2",
        "Transpose",
        "Where",
        "ZerosLike",
]
