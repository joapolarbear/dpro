class ncclGraph:
    def __init__(self):
        self.graph = {}

    def parse_tree_topo(self, tree_dict):
        ### "Tree": {"-1": "[0] 1/-1/-1->0->-1|-1->0->1/-1/-1 [1] 1/-1/-1->0->3|3->0->1/-1/-1"}
        if "Tree" not in self.graph:
            self.graph["Tree"] = {}
        tree_str = tree_dict["-1"]
        tree_split = tree_str.split("[")
        for i in range(1, len(tree_split)):
            channel_str = tree_split[i]
            channel_id = int(channel_str[0])
            if channel_id not in self.graph["Tree"]:
                self.graph["Tree"][channel_id] = {}
            channel_str_split = channel_str.split(" ")[1].split("->")
            ### rank
            assert channel_str_split[1] == channel_str_split[3]
            rank = int(channel_str_split[1])
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
            if rank not in self.graph["Tree"][channel_id]:
                self.graph["Tree"][channel_id][rank] = {}
            self.graph["Tree"][channel_id][rank]["down"] = down
            self.graph["Tree"][channel_id][rank]["up"] = up


    def parse_ring_topo(self, ring_dict):
        ### "1": "3[3000] -> 0[2000] [receive] via NET/Socket/0,0[2000] -> 1[3000] via direct shared memory,0[2000] -> 3[3000] [send] via NET/Socket/0",
        ### "0": "3[3000] -> 0[2000] [receive] via NET/Socket/0,0[2000] -> 1[3000] via direct shared memory"
        if "Ring" not in self.graph:
            self.graph["Ring"] = {}
        for channel_id_s, channel_str_s in ring_dict.items():
            channel_id = int(channel_id_s)
            if channel_id not in self.graph["Ring"]:
                self.graph["Ring"][channel_id] = {}
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
                if sender not in self.graph["Ring"][channel_id]:
                    self.graph["Ring"][channel_id][sender] = {"next": set(), "prev": set()}
                if recver not in self.graph["Ring"][channel_id]:
                    self.graph["Ring"][channel_id][recver] = {"next": set(), "prev": set()}
                self.graph["Ring"][channel_id][sender]["next"].add((recver, tspt_type))
                self.graph["Ring"][channel_id][recver]["prev"].add((sender, tspt_type))

    def print_graph(self):
        if "Tree" in self.graph:
            print("Algorithm: %s" % "Tree")
            for channel, channel_dict in sorted(self.graph["Tree"].items()):
                print("\tChannel: %d" % channel)
                for rank, rank_str in sorted(channel_dict.items()):
                    print("\t\tRank %d: child %s parent %d" % 
                        (rank, rank_str["down"], rank_str["up"]))

        if "Ring" in self.graph:
            print("Algorithm: %s" % "Ring")
            for channel, channel_dict in sorted(self.graph["Ring"].items()):
                print("\tChannel: %d" % channel)
                for rank, rank_str in sorted(channel_dict.items()):
                    print("\t\tRank %d: " % rank)
                    for peer_rank, t_type in rank_str["prev"]:
                    	print("\t\t\tRecv from %d via %s" % (peer_rank, t_type))
                    for peer_rank, t_type in rank_str["next"]:
                    	print("\t\t\tSend to %d via %s" % (peer_rank, t_type))






