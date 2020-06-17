from scapy.all import *
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
            if index < 4 or index > 24:
                return None
        key = int.from_bytes(data[index+4:index+12], "little", signed=False)
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
            if index > 24:
                return None
            key = int.from_bytes(data[index+4:index+12], "little", signed=False)
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

def __generate_comm_events(logs, pid, key_to_tensor_name):
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

def preprocess_pcap(pcap_paths, process_names_list, node_ip_to_rank, 
                    gradient_name_list_path, key_dict_path, 
                    save_path = None):
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

    # read id to name mapping
    key_to_tensor_name = {}
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
                if len(key_list) > 1:
                    for index, key in enumerate(key_list):
                        key_to_tensor_name[int(key)] = tensor_name+"~PART"+str(index)
                else:
                    key_to_tensor_name[int(key_list[0])] = tensor_name

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
        print(process_names_list[index])
        for key, logs in log_dict.items():
            _, ip = key
            events = __generate_comm_events(logs, key_to_pid[key], key_to_tensor_name)
            if '1' in process_names_list[index]:
                for ev in events:
                    if 'ts' in ev:
                        ev['ts'] = ev['ts']
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
        json.dump(chrome_json, f)
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

def parse_server_logs(server_log_paths, node_rank_list, gradient_name_list_path, 
                        key_dict_path, save_path = None):
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
                if len(key_list) > 1:
                    for index, key in enumerate(key_list):
                        key_to_tensor_name[int(key)] = tensor_name+"~PART"+str(index)
                else:
                    key_to_tensor_name[int(key_list[0])] = tensor_name
    
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
        json.dump(chrome_json, f)
        real_path = os.path.realpath(f.name)

    return real_path