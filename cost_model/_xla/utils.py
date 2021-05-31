class CMPaths:
    DATASET_DIR = "dataset"
    DEBUG_DIR = "debug"
    HLO_DIR = "hlos"
    PROFILE_DIR = "xla_profile"
    FEATURE_DIR = "features"
    MODULES_DIR = "modules"
    RAW_SUBGRAPH_DIR = "generated_subgraph"

    LABEL_FILE = "labels.txt"
    TF_SUPPORTED_OPS_FILE = "tf_xla_supported_ops.txt"

    METADATA_FILE = "metadata.json"
    RAW_GRAPH_DEF_FILE = "final_graph.json"
    CLEANED_GRAPH_DEF_FILE = "cleaned_graph.json"
    UNIQUE_OP_HISTORY_FILE = "unique_op_history.txt"

    MAX_CLUSTER_CACHE_FILE = "max_cluster.pickle"
    ELEMENTARY_OP_CACHE_FILE = "elementary_ops.txt"

    DATASET_SAVE_FILE = "dataset.pickle"
    ELEMENTARY_OP_CACHE_SAVE_FILE = "elem_op_cache.pickle"
    OVERHEAD_MODEL_SAVE_FILE = "overhead.pickle"
    MODEL_WEIGHT_SAVE_FILE = "model_weights.h5"
    MODEL_CONFIG_FILE = "model_config.pickle"
    MODULE_CONFIG_FILE = "module_config.txt"
    GRAPH_DEF_PICKLE_FILE = "graph_def.pickle"

    AFTER_OPT_TF_DAG_FILE = "partition_def_0.json"
    DEBUG_XLA_CANDIATES_FILE = "PLEASE SPECIFY CANDIDATE FILE PATH"
    TENSOR_SHAPE_FILE = "tensor_shapes.json"

class CMEnvs:
    WHITE_LIST_PATH = "BPF_XLA_OP_WHITE_LIST_PATH"
    TF_PATH = "BPF_TF_PATH"
    CM_PROFILE_GPU = "BPF_COST_MODEL_PROFILE_GPU"


### TODO(huhanpeng): ResourceApplyGradientDescent should not be ignored
IGNORE_OP_TYPES = ["ShapeN", "_Arg", "_Send", "_Recv", "VarIsInitializedOp", "ReadVariableOp", "VarHandleOp",
                   "IsVariableInitialized", "ResourceApplyGradientDescent", "IteratorToStringHandle",
                   "IteratorGetNext", "MakeIterator", "IteratorV2", "NoOp", "Placeholder"]


def parse_xla_candidate_ops(candidate_path):
    candidates = set()
    graph_node_id2name = {}
    unsafe_resource_deps_ = set()
    with open(candidate_path, "r") as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("unsafe_resource_deps_"):
            idx += 1
            break
        ls = lines[idx].strip().split(" ")
        candidates.add(ls[0])
        graph_node_id2name[ls[1]] = ls[0]
        idx += 1
    while idx < len(lines):
        ls = lines[idx].strip().split(" ")
        unsafe_resource_deps_.add(
            (graph_node_id2name[ls[0]], graph_node_id2name[ls[1]]))
        idx += 1
    return candidates, unsafe_resource_deps_
