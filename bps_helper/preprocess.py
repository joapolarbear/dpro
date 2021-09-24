from scapy.all import *
from scapy.layers.inet import TCP
from scapy.layers.inet import IP
from arg_utils import SingleArg
import json
import os

args_ = SingleArg().args

############################### PACKETS PARSING ################################

def __parse_single_packet(packet):
    time = float(packet.time)
    data = packet[TCP][Raw].load
    index_s = data.find(b"s:")
    index_e = data.find(b"e:")
    try:
        if index_s < 0:
            index = index_e
            selected = "e"
        elif index_e < 0:
            index = index_s
            selected = "s"
        else:
            if index_s < index_e:
                index = index_s
                selected = "s"
            else:
                index = index_e
                selected = "e"
        if selected == "s":
            if index < 4:
                index = data.find(b"s:", index+2)
                if index < 4 or index > 37:
                    return None
            key = int.from_bytes(data[index+4+8:index+12+8], "little", signed=False)
            is_request = data[index+2]
            if is_request != 1 and is_request != 0:
                return None
            is_push = data[index+3]
            if is_push != 0 and is_push != 1:
                return None
            if key != (1 << 64) -1:
                return ("start", bool(is_push), bool(is_request), time, key, packet[TCP].sport, packet[IP].dst, packet[TCP].seq)
            else:
                return None
        else:
            if index != -1:
                if index > 37:
                    return None
                key = int.from_bytes(data[index+4+8:index+12+8], "little", signed=False)
                is_request = data[index+2]
                if is_request != 1 and is_request != 0:
                    return None
                is_push = data[index+3]
                if is_push != 0 and is_push != 1:
                    return None
                if key != (1 << 64) -1:
                    return ("end", bool(is_push), bool(is_request), time, key, packet[TCP].sport, packet[IP].dst, packet[TCP].seq)
                else:
                    return None
            else:
                return None
    except:
        return None

def __separate_pair(parsed):
    port_dict = {}
    for se, op, req, timestamp, tid, sport, dport, seq in parsed:
        if (sport, dport) not in port_dict:
            port_dict[(sport, dport)] = []
        port_dict[(sport, dport)].append((se, "push" if op else "pull", "req" if req else "res",timestamp, tid, seq))
    return port_dict

def __remove_resend(log_to_remove):
    removed_packets = []
    sorted_packets = sorted(log_to_remove, key=lambda p: p[5])
    for index, p in enumerate(sorted_packets):
        if index != 0:
            if p[5] != sorted_packets[index-1][5]:
                removed_packets.append(p[:5])
        else:
            removed_packets.append(p[:5])
    return sorted(removed_packets, key=lambda p: p[3])

def __check_valid(clean_port_logs):
    for port_pair, logs in clean_port_logs.items():
        log_dict = {}
        log_len = len(logs)
        for i in range(log_len):
            if logs[i][0] == "start":
                if logs[i][4] in log_dict:
                    print(port_pair, i, logs[i])
                log_dict[logs[i][4]] = (i, logs[i])
            else:
                try:
                    del log_dict[logs[i][4]]
                except:
                    print(port_pair, i, logs[i])
        if len(log_dict) != 0:
            print(port_pair, log_dict)
            
def __parse_packets(packets, key_to_tensor_name):
    parsed = []
    for packet in packets:
        res = __parse_single_packet(packet)
        if res:
            _, _, _, _, tid, _, _, _ = res
            if tid in key_to_tensor_name:
                parsed.append(res)
    port_logs = __separate_pair(parsed)
    for key, log in port_logs.items():
        port_logs[key] = __remove_resend(log)
    # remove duplicates
    cleaned_port_logs = {}
    for port_pair in port_logs.keys():
        cleaned_port_logs[port_pair] = []
        for index, (se, op, req, time, tid) in enumerate(port_logs[port_pair]):
            if index == len(port_logs[port_pair])-1:
                cleaned_port_logs[port_pair].append((se, op, req, time, tid))
            else:
                next_se, next_op, next_req, next_time, next_tid = port_logs[port_pair][index+1]
                if op == next_op and se == next_se and req == next_req and tid == next_tid:
                    continue
                else:
                    cleaned_port_logs[port_pair].append((se, op, req, time, tid))
    __check_valid(cleaned_port_logs)
    return cleaned_port_logs

def __generate_comm_events_capture(logs, pid, key_to_tensor_name):
    events = []
    for ev, op, res, ts, key in logs:
        if key in key_to_tensor_name:
            event = {}
            event["name"] = key_to_tensor_name[key] + "."+op.upper()
            event["ph"] = "B" if ev == 'start' else "E"
            event["ts"] = ts * 1e6
            event["pid"] = pid
            event["tid"] = 0
            events.append(event)
        else:
            event = {}
            event["name"] = str(key) + "."+op.upper()
            event["ph"] = "B" if ev == 'start' else "E"
            event["ts"] = ts * 1e6
            event["pid"] = pid
            event["tid"] = 0
            events.append(event)
    return events

def __populate_key_name_mapping(key_dict_path, gradient_name_list_path = None, platform="TENSORFLOW"):
    key_to_tensor_name = {}
    if platform == "MXNET":
        # read id to name mapping
        tensor_index_to_tensor_name = {}
        with open(gradient_name_list_path, "r") as f:
            index = 0
            for line in f:
                tensor_index_to_tensor_name[index] = line.strip()
                index += 1

        with open(key_dict_path, "r") as f:
            for line in f:
                name, keys = line.split(":")
                if "gradient" in name:
                    tensor_id = int(name.split("_")[-1])
                    tensor_name = tensor_index_to_tensor_name[tensor_id]
                    key_list = keys.split()
                    try:
                        if len(key_list) > 1:
                            for index, key in enumerate(key_list):
                                key_to_tensor_name[int(key)] = tensor_name+"~PART"+str(index)
                        else:
                            key_to_tensor_name[int(key_list[0])] = tensor_name
                    except:
                        pass
    elif platform == "TENSORFLOW":
        with open(key_dict_path, "r") as f:
            for line in f:
                try:
                    name, keys = line.split(":")
                except:
                    continue
                if "BytePSPushPull" in name or "grad" in name:
                    tensor_name = name.strip()
                    tensor_name_split = tensor_name.split(".")
                    tensor_name = "{}.{}".format(tensor_name_split[0], tensor_name_split[1].replace("_", "+"))
                    key_list = keys.split()
                    try:
                        if len(key_list) > 1:
                            for index, key in enumerate(key_list):
                                key_to_tensor_name[int(key)] = tensor_name+"~PART"+str(index)
                        else:
                            key_to_tensor_name[int(key_list[0])] = tensor_name
                    except:
                        pass
    else:
        raise NotImplementedError("Unsupported platform {}.".format(platform))
    return key_to_tensor_name


def preprocess_pcap(pcap_paths, process_names_list, node_ip_to_rank, 
                    key_dict_path, gradient_name_list_path=None,
                    save_path=None, platform="TENSORFLOW"):
    """
    Reads and preprocesses captured pcap communication trace files. 

    Parameters
    ----------
    pcap_paths : list
        Paths to captured pcap files.

    process_names_list: list
        Process names of the corresponding pcap file. should be of the same 
        order as pcap_paths. Names should be in the format of "worker_" or 
        "server_" followed by node rank.
        e.g. ["worker_0", "worker_1", "server_0", "server_1"]
    
    node_ip_to_rank: dict
        IP address to node rank mappings.
        e.g. {'10.xx.x.12': 0, '10.xx.xx.13': 1}
    
    gradient_name_list_path: str
        Path to the file containing the list of gradient names. This is captured
        by the trace collecting process.
    
    key_dict_path: str
        Path to the file containing the gradient name to communication key 
        mapping. This is captured by the trace collecting process.

    save_path: str
        Path to the output trace file. Default: "PATH/comm_timeline.json"

    """
    # sanity check
    assert len(process_names_list) == len(pcap_paths)

    # get default save_path
    if save_path is None:
        save_path = os.path.join(args_.path, "comm_timeline.json")

    key_to_tensor_name = __populate_key_name_mapping(key_dict_path, gradient_name_list_path=gradient_name_list_path, platform=platform)

    # read pcap files and perform preprocessing
    machine_logs_list = []
    for path in pcap_paths:
        packets = rdpcap(path)
        logs = __parse_packets(packets, key_to_tensor_name)
        machine_logs_list.append(logs)

    pid_count = 0
    pid_to_name = {}
    key_to_pid = {}
    for index, machine_log in enumerate(machine_logs_list):
        for key in machine_log.keys():
            key_to_pid[key] = pid_count
            pid_to_name[pid_count] = str(process_names_list[index]) + " -> " + \
                                        ("server_" if process_names_list[index].split("_")[0] == "worker" else "worker_") + \
                                        str(node_ip_to_rank[key[1]])
            pid_count += 1

    chrome_events = []
    for index, log_dict in enumerate(machine_logs_list):
        # print(process_names_list[index])
        for key, logs in log_dict.items():
            events = __generate_comm_events_capture(logs, key_to_pid[key], key_to_tensor_name)
            chrome_events += events

    meta_events = []
    for pid in range(len(pid_to_name)):
        meta_events.append(
            {"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                "args": {
                "name" : pid_to_name[pid]
                }
            }
        )

    chrome_json = meta_events + chrome_events

    with open(save_path, 'w') as f:
        json.dump(chrome_json, f, indent=4)
        real_path = os.path.realpath(f.name)
    
    return real_path

############################# TIME STAMP PARSING ###############################

def __parse_timestamp_logs(log_lines, key_to_tensor_name):
    # parse header
    node_metas = []
    for line in log_lines:
        if line.startswith("role"):
            metas = [m.strip() for m in line.split(",")]
            node_meta = {}
            for meta in metas:
                key, val = meta.split("=")
                node_meta[key] = val
            node_metas.append(node_meta)

    logs = {}
    for line in log_lines:
        line = line.strip()
        if line.startswith("role"):
            continue
        try:
            ts, is_start, is_push, is_req, tid, sender, recver = line.split()
        except:
            print("preprocess warning: {}".format(line))
            continue
        tid = int(tid)
        ts = int(ts)
        sender = int(sender)
        recver = int(recver)
        if is_start in ["True", "true"]:
            is_start = True
        else:
            is_start = False
        if is_push in ["True", "true"]:
            is_push = True
        else:
            is_push = False
        if is_req in ["True", "true"]:
            is_req = True
        else:
            is_req = False
        if tid in key_to_tensor_name:
            if (sender, recver) not in logs:
                logs[(sender, recver)] = []
            logs[(sender, recver)].append((ts, is_start, is_push, is_req, tid))
        # else:
        #     print("Sender: {}, Recver: {}, ts: {}, is_start: {}, is_push: {}, is_req: {}, tid: {}".format(sender, recver, ts, is_start, is_push, is_req, tid))
        #     input()
    return logs, node_metas

def __generate_comm_events_timestamp(logs, pid, key_to_tensor_name):
    events = []
    for (ts, is_start, is_push, _, tid) in logs:
        op = "PUSH" if is_push else "PULL"
        if tid in key_to_tensor_name:
            event = {}
            event["name"] = key_to_tensor_name[tid] + "." + op
            event["ph"] = "B" if is_start else "E"
            event["ts"] = ts
            event["pid"] = pid
            event["tid"] = 0
            events.append(event)
        else:
            event = {}
            event["name"] = str(tid) + "."+ op
            event["ph"] = "B" if is_start else "E"
            event["ts"] = ts
            event["pid"] = pid
            event["tid"] = 0
            events.append(event)
    return events

def preprocess_comm_timestamp(file_paths, 
                            key_dict_path, gradient_name_list_path=None,
                            save_path=None, platform="TENSORFLOW"):
    """
    Reads and preprocesses communication timestamp files. 

    Parameters
    ----------
    file_paths : list
        Paths to timestamp files.
    
    gradient_name_list_path: str
        Path to the file containing the list of gradient names. This is captured
        by the trace collecting process.
    
    key_dict_path: str
        Path to the file containing the gradient name to communication key 
        mapping. This is captured by the trace collecting process.

    save_path: str
        Path to the output trace file. Default: "PATH/comm_timeline.json"

    """
    # get default save_path
    if save_path is None:
        save_path = os.path.join(args_.path, "comm_timeline.json")

    tid_to_tensor_name = __populate_key_name_mapping(key_dict_path, gradient_name_list_path=gradient_name_list_path, platform=platform)

    # read pcap files and perform preprocessing
    logs_dict = {}
    all_node_metas = []
    for path in file_paths:
        with open(path, "r") as f:
            lines = f.read().splitlines()
            logs, node_metas = __parse_timestamp_logs(lines, tid_to_tensor_name)
        for key, val in logs.items():
            if key not in logs_dict:
                logs_dict[key] = val
            else:
                logs_dict[key] += val
        
        node_rank = int(os.path.basename(path).split(".")[0].split("_")[1])
        for node_meta in node_metas:
            node_meta["node_rank_v2"] = node_rank

        all_node_metas += node_metas
    
    node_id_to_name = {}
    for node_meta in all_node_metas:
        nid = int(node_meta["id"])
        role = node_meta["role"]
        node_rank = node_meta["node_rank_v2"]
        node_name = role + "_" + str(node_rank)
        node_id_to_name[nid] = node_name

    pid_count = 0
    pid_to_name = {}
    key_to_pid = {}
    for (sender, recver) in logs_dict.keys():
        if sender in node_id_to_name and recver in node_id_to_name:
            key_to_pid[(sender, recver)] = pid_count
            pid_to_name[pid_count] = node_id_to_name[sender] + " -> " + node_id_to_name[recver]
            pid_count += 1

    chrome_events = []
    for (sender, recver), logs in logs_dict.items():
        if (sender, recver) in key_to_pid:
            events = __generate_comm_events_timestamp(logs, key_to_pid[(sender, recver)], tid_to_tensor_name)
            chrome_events += events

    meta_events = []
    for pid in range(len(pid_to_name)):
        meta_events.append(
            {"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                "args": {
                "name" : pid_to_name[pid]
                }
            }
        )

    chrome_json = meta_events + chrome_events

    with open(save_path, 'w') as f:
        json.dump(chrome_json, f, indent=4)
        real_path = os.path.realpath(f.name)
    
    return real_path
############################# SERVER LOG PARSING ###############################

def __parse_server_log(line):
    time, info = line.split(":")
    splitted = info.split(",")
    if len(splitted) == 3:
        return (int(time), splitted[0].strip(), int(splitted[1].strip()), splitted[2].strip(), 0)
    elif len(splitted) == 4:
        return (int(time), splitted[0].strip(), int(splitted[1].strip()), splitted[2].strip(), int(splitted[3].strip())+1)

def __generate_server_events(logs, pid, key_to_tensor_name):
    events = []
    for ts, ph, key, op, tid in logs:
        if key in key_to_tensor_name and op.upper() != "INIT":
            event = {}
            event["name"] = key_to_tensor_name[key] + "." + op.upper()
            event["ph"] = "B" if ph == 'start' else "E"
            event["ts"] = ts
            event["pid"] = pid
            event["tid"] = tid
            events.append(event)
    return events

def parse_server_logs(server_log_paths, node_rank_list, key_dict_path,
                        gradient_name_list_path, save_path = None, platform="TENSORFLOW"):
    """
    Reads and preprocesses captured server trace files. 

    Parameters
    ----------
    server_log_paths : list
        Paths to captured server trace files.

    node_rank_list: list
        Node ranks of the corresponding pcap file. should be of the same 
        order as server_log_paths.
        e.g. [0, 1, 2, 3]
    
    gradient_name_list_path: str
        Path to the file containing the list of gradient names. This is captured
        by the trace collecting process.
    
    key_dict_path: str
        Path to the file containing the gradient name to communication key 
        mapping. This is captured by the trace collecting process.

    save_path: str
        Path to the output trace file. Default: "PATH/server_timeline.json"

    """
    # get default save_path
    if save_path is None:
        save_path = os.path.join(args_.path, "server_timeline.json")

    # read name mappings
    key_to_tensor_name = {}
    if platform == "MXNET":
        tensor_index_to_tensor_name = {}
        with open(gradient_name_list_path, "r") as f:
            index = 0
            for line in f:
                tensor_index_to_tensor_name[index] = line.strip()
                index += 1

        with open(key_dict_path, "r") as f:
            for line in f:
                name, keys = line.split(":")
                if "gradient" in name:
                    tensor_id = int(name.split("_")[-1])
                    tensor_name = tensor_index_to_tensor_name[tensor_id]
                    key_list = keys.split()
                    try:
                        if len(key_list) > 1:
                            for index, key in enumerate(key_list):
                                key_to_tensor_name[int(key)] = tensor_name+"~PART"+str(index)
                        else:
                            key_to_tensor_name[int(key_list[0])] = tensor_name
                    except:
                        pass
    elif platform == "TENSORFLOW":
        with open(key_dict_path, "r") as f:
            for line in f:
                try:
                    name, keys = line.split(":")
                except:
                    continue
                if "BytePSPushPull" in name or "grad" in name:
                    tensor_name = name.strip()
                    tensor_name_split = tensor_name.split(".")
                    tensor_name = "{}.{}".format(tensor_name_split[0], tensor_name_split[1].replace("_", "+"))
                    key_list = keys.split()
                    try:
                        if len(key_list) > 1:
                            for index, key in enumerate(key_list):
                                key_to_tensor_name[int(key)] = tensor_name+"~PART"+str(index)
                        else:
                            key_to_tensor_name[int(key_list[0])] = tensor_name
                    except:
                        pass
    else:
        raise NotImplementedError("Unsupported platform {}.".format(platform))
    
    # read server logs
    server_logs = []
    for path in server_log_paths:
        log = []
        with open(path, "r") as f:
            for line in f:
                log.append(__parse_server_log(line))
        server_logs.append(log)

    chrome_events = []
    for index, log in enumerate(server_logs):
        chrome_events += __generate_server_events(log, int(node_rank_list[index]), key_to_tensor_name)

    meta_events = []

    def pid_to_name(pid):
        return "server_" + str(pid)

    for pid in node_rank_list:
        meta_events.append(
            {"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                "args": {
                "name" : pid_to_name(pid)
                }
            }
        )
    
    chrome_json = meta_events + chrome_events

    with open(save_path, 'w') as f:
        json.dump(chrome_json, f, indent=4)
        real_path = os.path.realpath(f.name)

    return real_path