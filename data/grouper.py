from amp_cost_model import CurveFiter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Grouper:
    def __init__(self, td_len=0.1, fd_len=0.01, unit_len=0.001, max_grp_size=20):
        self.fitter_table = {}

        assert td_len <= 1 and fd_len <= fd_len, "td_len ({}) must be <= 1 and fd_len ({}) must be <= td_len".format(td_len, fd_len)
        # assert fd_len == 0 or td_len % fd_len == 0, "td_len ({}) must be divisible by fd_len ({}) but {} or fd_len = 0".format(td_len, fd_len, td_len % fd_len)
        # assert 1 % td_len == 0, "1 must be divisible by td_len ({})".format(td_len)
        self.td_len = td_len
        self.fd_len = fd_len

        ### unit_len is used since the target dimension may not be integer
        self.unit_len = unit_len

        self.max_grp_size = max_grp_size

    def divide_with_upper(self, target_dim, train_x, train_y, test_x, test_y, headers, op_type="conv", visual=True):
        if target_dim != 'avg' and target_dim != 'intensity':
            target_idx = headers.index(target_dim) - 1

        sort_ind = np.argsort(train_x[:, target_idx])
        grp_id = 0
        grp_size = 0
        for i in sort_ind:
            if grp_id not in self.fitter_table:
                self.fitter_table[grp_id] = {
                    "train_x": [], "train_y": [],
                    "test_x": [], "test_y": []}
            self.fitter_table[grp_id]["train_x"].append(train_x[i])
            self.fitter_table[grp_id]["train_y"].append(train_y[i])

            ### (TODO) huhanpeng, test data should be not training data
            self.fitter_table[grp_id]["test_x"].append(train_x[i])
            self.fitter_table[grp_id]["test_y"].append(train_y[i])

            grp_size += 1
            if grp_size >= self.max_grp_size:
                grp_id += 1
                grp_size = 0


    def divide_by_len(self, target_dim, train_x, train_y, test_x, test_y, headers):
        """
        Parameters
        ----------
        target_dim: traget dimension used for division
        train_x : shape = (n_samples, n_features)
        ...
        """
        ### target dimension index needs to exclude avg
        if target_dim != 'avg' and target_dim != 'intensity':
            target_idx = headers.index(target_dim) - 1

        # self.unit_len = min(min(train_x[target_idx, :]), min(test_x[target_idx, :]))

        for i in range(train_x.shape[0]):
            if target_dim == 'avg':
                x = train_y[i, 0]
            elif target_dim == 'intensity':
                raise
            elif target_dim == 'B':
                x = train_x[i, target_idx]
            else:
                x = train_x[i, target_idx]
            grp_id = "%03d_%03d"%(int(x / self.td_len), int((x % self.td_len) % self.fd_len / self.unit_len)) \
                if self.fd_len > 0 else "%03d_%03d"%(int(x / self.td_len), 0)
            if grp_id not in self.fitter_table:
                self.fitter_table[grp_id] = {
                    "train_x": [], "train_y": [],
                    "test_x": [], "test_y": []}
            self.fitter_table[grp_id]["train_x"].append(train_x[i])
            self.fitter_table[grp_id]["train_y"].append(train_y[i])

        for i in range(test_x.shape[0]):
            if target_dim == 'avg':
                x = test_y[i, 0]
            elif target_dim == 'intensity':
                raise
            else:
                x = test_x[i, target_idx]
            grp_id = "%03d_%03d"%(int(x / self.td_len), int((x % self.td_len) % self.fd_len / self.unit_len)) \
                if self.fd_len > 0 else "%03d_%03d"%(int(x / self.td_len), 0)
            if grp_id not in self.fitter_table:
                self.fitter_table[grp_id] = {
                    "train_x": [], "train_y": [],
                    "test_x": [], "test_y": []
                }
            self.fitter_table[grp_id]["test_x"].append(test_x[i])
            self.fitter_table[grp_id]["test_y"].append(test_y[i])

    def train_test(self, target_dim, headers, op_type="conv", visual=True):
        print("Total number of groups: {}".format(len(self.fitter_table)))
        plots = [[], [], [], []]
        for grp_id in sorted(self.fitter_table.keys()):
            self.fitter_table[grp_id]["fitter"] = CurveFiter(
                    np.array(self.fitter_table[grp_id]["train_x"]), 
                    np.array(self.fitter_table[grp_id]["train_y"]), 
                    np.array(self.fitter_table[grp_id]["test_x"]), 
                    np.array(self.fitter_table[grp_id]["test_y"]), 
                    headers, op_type=op_type
                    )
            print("Group ID {} collects training data - X:{}, Y:{}, test data - X:{}, Y:{}".format(
                    grp_id,
                    self.fitter_table[grp_id]["fitter"].train_x.shape, 
                    self.fitter_table[grp_id]["fitter"].train_y.shape, 
                    self.fitter_table[grp_id]["fitter"].test_x.shape, 
                    self.fitter_table[grp_id]["fitter"].test_y.shape))
            popt, pcov = self.fitter_table[grp_id]["fitter"].train()
            error = self.fitter_table[grp_id]["fitter"].test(verbose=True)

            if error is not None:
                plots[0].append(grp_id)
                plots[1].append(self.fitter_table[grp_id]["fitter"].train_x.shape[0])
                plots[2].append(self.fitter_table[grp_id]["fitter"].test_x.shape[0])
                plots[3].append(error)

        if visual:
            plt.figure(num=1, figsize=(8, 6))
            clrs = sns.color_palette("husl", 5)
            xaxis = np.arange(len(plots[0]))
            bar_width = 0.4

            fig, ax= plt.subplots()
            ax.set_xlabel(target_dim)
            ax.bar(xaxis, plots[1], label='training data size', width=bar_width, color=clrs[0])
            ax.bar(xaxis + bar_width, plots[2], label='test data size', width=bar_width, color=clrs[2])
            plt.xticks(xaxis + bar_width/2, plots[0], rotation=80)
            plt.legend(loc=1)
            
            ax = ax.twinx()
            ax.plot(plots[0], plots[3], '.-', label='fitting error = %6.4f %%'%(sum(plots[3])/len(plots[3])))
            ax.set_ylabel("Fitting error (%)")
            plt.legend(loc=2)

            ax.tick_params(axis='y', labelcolor=clrs[3])
            plt.title("Data size and fitting error of each group \n (td_len={}, fd_len={}, unit_len={})".format(self.td_len, self.fd_len, self.unit_len))

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()


            






