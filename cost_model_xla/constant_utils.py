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
    GRAPH_DEF_PICKLE_FILE = "graph_def.pickle"

    AFTER_OPT_TF_DAG_FILE = "partition_def_0.json"
    DEBUG_XLA_CANDIATES_FILE = "/root/xla_candidates_resnet.txt"
    TENSOR_SHAPE_FILE = "tensor_shapes.json"

class CMEnvs:
    WHITE_LIST_PATH = "BPF_XLA_OP_WHITE_LIST_PATH"
    TF_PATH = "BPF_TF_PATH"
    CM_PROFILE_GPU = "BPF_COST_MODEL_PROFILE_GPU"