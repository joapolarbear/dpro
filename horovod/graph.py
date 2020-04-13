from enum import Enum
class NCCL_ALGO(Enum):
    TREE=0
    RING=1

class ncclGraph:
    def __init__(self):
        '''
        One example of the graph is shown below
        Algorithm: Tree
            Channel: 0
                Rank 0: child [1] parent -1
                Rank 1: child [2] parent 0
                Rank 2: child [3] parent 1
                Rank 3: child [] parent 2
            Channel: 1
                Rank 0: child [1] parent 3
                Rank 1: child [] parent 0
                Rank 2: child [3] parent -1
                Rank 3: child [0] parent 2
        Algorithm: Ring
            Channel: 0
                Rank 0: 
                    Recv from 3 via Socket
                    Recv from 1 via SHM
                    Send to 1 via SHM
                Rank 1: 
                    Recv from 2 via Socket
                    Recv from 0 via SHM
                    Send to 2 via Socket
                    Send to 0 via SHM
                Rank 2: 
                    Recv from 3 via SHM
                    Recv from 1 via Socket
                    Send to 3 via SHM
                    Send to 1 via Socket
                Rank 3: 
                    Recv from 2 via SHM
                    Send to 0 via Socket
                    Send to 2 via SHM
            Channel: 1
                Rank 0: 
                    Recv from 3 via Socket
                    Recv from 1 via SHM
                    Send to 1 via SHM
                    Send to 3 via Socket
                Rank 1: 
                    Recv from 0 via SHM
                    Send to 2 via Socket
                    Send to 0 via SHM
                Rank 2: 
                    Recv from 3 via SHM
                    Recv from 1 via Socket
                    Send to 3 via SHM
                Rank 3: 
                    Recv from 0 via Socket
                    Recv from 2 via SHM
                    Send to 0 via Socket
                    Send to 2 via SHM
        '''
        
        self.algo = None
        self.rank_num = 0
        ### Set freeze to true if the graph will not change
        self.freeze = False
        self.trace_parsed = False

        self.graph = {}
        self.raw_name2IDnum = {}
        self.rank2prefix = {}
        self.prefix2rank = {}

       
    def parse_tree_topo(self, tree_dict, map_to=None):
        ''' If `map_to` is set, map the rank to the given prefix '''
        ### "Tree": {"-1": "[0] 1/-1/-1->0->-1|-1->0->1/-1/-1 [1] 1/-1/-1->0->3|3->0->1/-1/-1"}
        self.algo = NCCL_ALGO.TREE
        if "Tree" not in self.graph:
            self.graph["Tree"] = {}
        tree_str = tree_dict["-1"]
        tree_split = tree_str.split("[")
        rank = None
        for i in range(1, len(tree_split)):
            channel_str = tree_split[i]
            channel_id = int(channel_str[0])
            if channel_id not in self.graph["Tree"]:
                self.graph["Tree"][channel_id] = {}
            channel_str_split = channel_str.split(" ")[1].split("->")

            ### Get the rank and map the rank to a prefix, ensure it only run once
            assert channel_str_split[1] == channel_str_split[3]
            if rank is None:
                rank = int(channel_str_split[1])
                if map_to is not None:
                    self.map_rank2prefix(rank, map_to)
                self.rank_num = max(self.rank_num, rank+1)
   
            ### down
            assert channel_str_split[0] == channel_str_split[4]
            down = []
            for c in channel_str_split[0].split("/"):
                if c != "-1":
                    down.append(int(c))
            ### up
            uplist = channel_str_split[2].split("|")
            assert len(uplist) == 2 and uplist[0] == uplist[1]
            up = int(uplist[0])

            ### Add up and down info the the graph
            if rank not in self.graph["Tree"][channel_id]:
                self.graph["Tree"][channel_id][rank] = {}
            self.graph["Tree"][channel_id][rank]["down"] = down
            self.graph["Tree"][channel_id][rank]["up"] = up


    def parse_connect_topo(self, ring_dict, map_to=None):
        ''' If `map_to` is set, map the rank to the given prefix '''
        ### "1": "3[3000] -> 0[2000] [receive] via NET/Socket/0,0[2000] -> 1[3000] via direct shared memory,0[2000] -> 3[3000] [send] via NET/Socket/0",
        ### "0": "3[3000] -> 0[2000] [receive] via NET/Socket/0,0[2000] -> 1[3000] via direct shared memory"
        self.algo = NCCL_ALGO.RING
        print(ring_dict)
        if "RingConnect" not in self.graph:
            self.graph["RingConnect"] = {}
        rank = None
        for channel_id_s, channel_str_s in ring_dict.items():
            channel_id = int(channel_id_s)
            if channel_id not in self.graph["RingConnect"]:
                self.graph["RingConnect"][channel_id] = {}
            for channel_str in channel_str_s.split(","):
                ### Retrive transport type
                if "Socket" in channel_str:
                    tspt_type = "Socket"
                elif "IB" in channel_str:
                    tspt_type = "IB"
                elif "P2P" in channel_str:
                    tspt_type = "P2P"
                elif "direct shared memory" in channel_str:
                    tspt_type = "SHM"

                channel_str_split = channel_str.split("[")

                sender = int(channel_str_split[0])
                recver = int(channel_str_split[1].split("->")[1])

                ### map rank to a prefix, ensure it only run once
                if rank is None:
                    rank = recver if "receive" in channel_str else sender
                    if map_to is not None:
                        self.map_rank2prefix(rank, map_to)
                    self.rank_num = max(self.rank_num, rank+1)

                if sender not in self.graph["RingConnect"][channel_id]:
                    self.graph["RingConnect"][channel_id][sender] = {"next": set(), "prev": set()}
                if recver not in self.graph["RingConnect"][channel_id]:
                    self.graph["RingConnect"][channel_id][recver] = {"next": set(), "prev": set()}
                self.graph["RingConnect"][channel_id][sender]["next"].add((recver, tspt_type))
                self.graph["RingConnect"][channel_id][recver]["prev"].add((sender, tspt_type))

    def parse_ring_topo(self, ring_dict, map_to=None):
        ''' If `map_to` is set, map the rank to the given prefix '''
        ### "1": "3[3000] -> 0[2000] [receive] via NET/Socket/0,0[2000] -> 1[3000] via direct shared memory,0[2000] -> 3[3000] [send] via NET/Socket/0",
        ### "0": "3[3000] -> 0[2000] [receive] via NET/Socket/0,0[2000] -> 1[3000] via direct shared memory"
        self.algo = NCCL_ALGO.RING
        if "Ring" not in self.graph:
            self.graph["Ring"] = {}
        rank = None
        for channel_id_s, channel_str_s in ring_dict.items():
            channel_id = int(channel_id_s)
            if channel_id not in self.graph["Ring"]:
                self.graph["Ring"][channel_id] = {}
            for channel_str in channel_str_s.split(","):
                channel_str_split = channel_str.split("[")

                sender = int(channel_str_split[0])
                recver = int(channel_str_split[1].split("->")[1])

                ### map rank to a prefix, ensure it only run once
                if rank is None:
                    rank = recver if "receive" in channel_str else sender
                    if map_to is not None:
                        self.map_rank2prefix(rank, map_to)
                    self.rank_num = max(self.rank_num, rank+1)

                if sender not in self.graph["Ring"][channel_id]:
                    self.graph["Ring"][channel_id][sender] = {"next": None, "prev": None}
                if recver not in self.graph["Ring"][channel_id]:
                    self.graph["Ring"][channel_id][recver] = {"next": None, "prev": None}
                self.graph["Ring"][channel_id][sender]["next"] = recver
                self.graph["Ring"][channel_id][recver]["prev"] = sender


    def map_rank2prefix(self, rank, prefix):
        if rank not in self.rank2prefix:
            self.rank2prefix[rank] = prefix
        if prefix not in self.prefix2rank:
            self.prefix2rank[prefix] = rank


    def print_graph(self):
        if not self.freeze:
            self.freeze_graph()

        if "Tree" in self.graph:
            print("Algorithm: %s" % "Tree")
            for channel, channel_dict in sorted(self.graph["Tree"].items()):
                print("\tChannel: %d" % channel)
                for rank, rank_dict in sorted(channel_dict.items()):
                    print("\t\tRank %d: child %s parent %d" % 
                        (rank, rank_dict["down"], rank_dict["up"]))

        if "RingConnect" in self.graph:
            print("Connection: %s" % "RingConnect")
            for channel, channel_dict in sorted(self.graph["RingConnect"].items()):
                print("\tChannel: %d" % channel)
                for rank, rank_dict in sorted(channel_dict.items()):
                    print("\t\tRank %d: " % rank)
                    for peer_rank, t_type in rank_dict["prev"]:
                        print("\t\t\tRecv from %d via %s" % (peer_rank, t_type))
                    for peer_rank, t_type in rank_dict["next"]:
                        print("\t\t\tSend to %d via %s" % (peer_rank, t_type))

        if "Ring" in self.graph:
            print("Algorithm: %s" % "Ring")
            for channel, channel_dict in sorted(self.graph["Ring"].items()):
                print("\tChannel: %d" % channel)
                for rank, rank_dict in sorted(channel_dict.items()):
                    print("\t\tRank %d: " % rank)
                    if "prev" in rank_dict:
                        peer_rank = rank_dict["prev"]
                        print("\t\t\tRecv from %d" % peer_rank)
                    if "next" in rank_dict:
                        peer_rank = rank_dict["next"]
                        print("\t\t\tSend to %d" % peer_rank)


    def parse_traces(self, traces):
        ''' `traces` must be sorted according to `ts` '''
        if self.trace_parsed:
        	return

        self.trace_parsed = True
        first_fwd = None
        if self.algo == NCCL_ALGO.RING or self.algo == NCCL_ALGO.TREE:
            for trace in traces:
                ### Stop the loop early if one FW trace appears twice
                if "FW" in trace["name"]:
                    if first_fwd is None:
                        first_fwd = trace["name"]
                    elif trace["name"] == first_fwd:
                        break

                ### Just check traces whose pid is comm_detail
                if "comm_detail" not in trace["tid"] and "comm_detail" not in trace["pid"]:
                    continue

                ### Get the rawname withoud RECV/SEND
                if ".RECV" in trace["name"]:
                	raw_name = trace["name"].split(".RECV")[0]
                elif ".SEND" in trace["name"]:
                	raw_name = trace["name"].split(".SEND")[0]
                else:
                	raw_name = trace["name"]

                if raw_name not in self.raw_name2IDnum:
                    self.raw_name2IDnum[raw_name] = {"chunkNum": 0, "sliceNum": 0, "channelNum": 0}
                self.raw_name2IDnum[raw_name]["chunkNum"] = max(int(trace["args"]["chunkId"]) + 1, self.raw_name2IDnum[raw_name]["chunkNum"])
                self.raw_name2IDnum[raw_name]["sliceNum"] = max(int(trace["args"]["sliceId"]) + 1, self.raw_name2IDnum[raw_name]["sliceNum"])
                self.raw_name2IDnum[raw_name]["channelNum"] = max(int(trace["args"]["channelId"]) + 1, self.raw_name2IDnum[raw_name]["channelNum"])
        else:
            raise ValueError("NCCL_ALGO error: %s" % self.algo.value)


    def get_IDnum(self, raw_name):
        assert self.trace_parsed is True
        return self.raw_name2IDnum[raw_name]["chunkNum"], self.raw_name2IDnum[raw_name]["sliceNum"], self.raw_name2IDnum[raw_name]["channelNum"]

    def freeze_graph(self):
        assert self.freeze is False
        self.freeze = True

    def nccl_dependency(self, prefix, chunkId, sliceId, channelId):
        '''
        Return
        ------
        `next rank prefix`, `next chunkId`, `next sliceId`, `next channelId`, 
        if these are all None, return a list of prefixes, to denote that this is the first communication operations
        it depends on all the BW/Negotiate nodes.
        if all of above variables are None, this is the last step
        '''
        if not self.freeze:
            self.freeze_graph()
        if self.algo == NCCL_ALGO.RING:
            ### For ring, chunkId denotes the step of each chunk of tensor
            if chunkId == 0:
                ### for the first chunk to be sent, it depends on the BW nodes of all ranks
                all_prefix = [self.rank2prefix[r] for r in sorted(self.graph["Ring"][channelId].keys())]
                return None, None, None, None, all_prefix
            if chunkId >= 2 * (self.rank_num - 1):
                ### the last step
                return None, None, None, None, None

            ### for the remaining steps
            rank = self.prefix2rank[prefix]
            c, k = self.ring_step_to_chunk_order_id(rank, chunkId, Send=True)
            ### !!! dependency
            next_rank = self.graph["Ring"][channelId][rank]["next"]
            nc = c
            nk = k
            next_chunkId = self.ring_chunk_order_id_to_step(next_rank, nc, nk, Send=False)
            
            return self.rank2prefix[next_rank], next_chunkId, sliceId, channelId, None

        elif self.algo == NCCL_ALGO.TREE:
            # TODO (huhanpeng)
            raise NotImplementedError()
        else:
            raise ValueError("NCCL_ALGO error: %s" % self.algo.value)


    def ring_step_to_chunk_order_id(self, p, t, Send=True):
        '''
        p is the process number
        t is the step, we use chunkId in this programe

        Return
        ------
        c is the chunk id sorted by the order of chunks
        k is the number of times one chunk has been processed
        '''
        if Send:
            c = (p - t + self.rank_num) % self.rank_num
        else:
            c = (p - t + self.rank_num - 1) % self.rank_num
        k = 1 if t >= self.rank_num else 0
        return c, k

    def ring_chunk_order_id_to_step(self, p, c, k, Send=True):
        '''
        c is the chunk id sorted by the order of chunks
        k is the number of times one chunk has been processed 

        Return
        ------
        p is the process number
        t is the step, we use chunkId in this programe
        '''
        if Send:
            if c <= p:
                t = p - c + k * self.rank_num
            else:
                t = p - c + (k + 1) * self.rank_num
        else:
            if c < p:
                t = p - 1 - c + k * self.rank_num
            else:
                t = p - 1 - c + (k + 1) * self.rank_num
        return t



