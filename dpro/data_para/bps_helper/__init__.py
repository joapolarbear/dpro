import os
from ...trace_utils import FileName, SingleLogger, SYNC_MODE

from .preprocess import preprocess_comm_timestamp, parse_server_logs
from .graph import bytepsGraph


def collect_bps_graph(force_, pm, byteps_graph,
        platform, van_type, zmq_log_path,
        server_log_path):
    byteps_cache_path = pm.search(FileName.BYTEPS_CACHE)
    if byteps_cache_path is not None and not force_:
        SingleLogger().info("Inited BytePS graph helper from cache: {}.".format(byteps_cache_path))
        byteps_graph.init_from_cache(byteps_cache_path)
    else:
        SingleLogger().info("Unable to find BytePS cache file.")
        # read or generate BPS comm_trace
        byteps_comm_detail_path = pm.search(FileName.BPS_COMM_DETAIL)
        if byteps_comm_detail_path is None or force_:
            # need to run preprocessing
            
            if platform == "MXNET":
                gradient_name_list_path = pm.search(FileName.TENSOR_NAME)
            else:
                gradient_name_list_path = None

            key_dict_path = pm.search(FileName.KEY_DICT)

            if van_type.upper() == "ZMQ":
                assert zmq_log_path is not None
                zmq_log_fns = [fn for fn in os.listdir(zmq_log_path) if (os.path.isfile(os.path.join(zmq_log_path,fn)) and fn.endswith(".log"))]
                zmq_log_paths = [os.path.join(zmq_log_path, fn) for fn in zmq_log_fns]
                SingleLogger().info("Preprocessing ZMQ log files: {}.".format(zmq_log_paths))
                byteps_comm_detail_path = preprocess_comm_timestamp(zmq_log_paths, key_dict_path, 
                    gradient_name_list_path=gradient_name_list_path, 
                    platform=platform,
                    save_path=os.path.join(pm.path, "comm_timeline.json"))
            elif van_type.upper() == "RDMA":
                assert zmq_log_path is not None
                zmq_log_fns = [fn for fn in os.listdir(zmq_log_path) if (os.path.isfile(os.path.join(zmq_log_path, fn)) and fn.startswith("rdma_"))]
                zmq_log_paths = [os.path.join(zmq_log_path, fn) for fn in zmq_log_fns]
                SingleLogger().info("Preprocessing RDMA log files: {}.".format(zmq_log_paths))
                byteps_comm_detail_path = preprocess_comm_timestamp(zmq_log_paths, key_dict_path,
                    gradient_name_list_path=gradient_name_list_path,
                    platform=platform,
                    save_path=os.path.join(pm.path, "comm_timeline.json"))
            else:
                raise ValueError("Invalide VAN type: {}".format(van_type))
        else:
            SingleLogger().info("Found BytePS comm trace file in {}.".format(byteps_comm_detail_path))
        # read or generate BPS server trace
        byteps_server_trace_path = pm.search(FileName.BPS_SERVER_TRACE)
        if byteps_server_trace_path is None or force_:
            # need to run preprocessing
            if server_log_path is None:
                SingleLogger().error("Cannot find BytePS server trace or raw log files.")
                exit(1)
            log_fns = [fn for fn in os.listdir(server_log_path) if os.path.isfile(os.path.join(server_log_path,fn)) and fn.endswith(".txt")]
            log_paths = [os.path.join(server_log_path, fn) for fn in log_fns]
            node_ranks = [int(fn.split(".txt")[0].split("_")[-1]) for fn in log_fns]
            if platform == "MXNET":
                gradient_name_list_path = pm.search(FileName.TENSOR_NAME)
            else:
                gradient_name_list_path = None
            key_dict_path = pm.search(FileName.KEY_DICT)
            SingleLogger().info("Parsing server log files: {}.".format(log_paths))
            byteps_server_trace_path = parse_server_logs(log_paths, node_ranks, key_dict_path, 
                gradient_name_list_path=gradient_name_list_path, 
                platform=platform,
                save_path=os.path.join(pm.path, "server_timeline.json"))
        else:
            SingleLogger().info("Found BytePS server trace file in {}".format(byteps_server_trace_path))
        # initialize BytePS graph helper
        byteps_graph.init(
            byteps_comm_detail_path,
            byteps_server_trace_path,
            pm.path,
            van_type=van_type,
            align_trace=(SYNC_MODE >= 0))
        