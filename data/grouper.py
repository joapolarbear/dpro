from amp_cost_model import CurveFiter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Delimiter:
    def __init__(self, target_dim, td_len=1., fd_len=0., unit_len=1., max_grp_size=20):
        assert td_len <= 1 and fd_len <= fd_len, "td_len ({}) must be <= 1 and fd_len ({}) must be <= td_len".format(td_len, fd_len)
        # assert fd_len == 0 or td_len % fd_len == 0, "td_len ({}) must be divisible by fd_len ({}) but {} or fd_len = 0".format(td_len, fd_len, td_len % fd_len)
        # assert 1 % td_len == 0, "1 must be divisible by td_len ({})".format(td_len)
        self.target_dim = target_dim
        self.td_len = td_len
        self.fd_len = fd_len
        ### unit_len is used since the target dimension may not be integer
        self.unit_len = unit_len
        self.max_grp_size = max_grp_size

class Grouper:
    def __init__(self, ):
        self.fitter_table = {}  

    def divide_with_upper(self, dels, train_x, train_y, test_x, test_y, headers, op_type="conv", visual=True):
        if dels.target_dim != 'avg' and dels.target_dim != 'intensity':
            target_idx = headers.index(dels.target_dim) - 1

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
            if grp_size >= dels.max_grp_size:
                grp_id += 1
                grp_size = 0

    def divide_by_len(self, dels, train_x, train_y, test_x, test_y, headers):
        """
        Parameters
        ----------
        dels: class `Delimiter` or a list of class `Delimiter`, containing traget dimension, fd_len, ... used for division
        train_x : shape = (n_samples, n_features)
        ...
        """
        if not isinstance(dels, list):
            dels = [dels]

        for i in range(train_x.shape[0]):
            grp_id = self.gen_grp_id(dels, train_x[i], train_y[i], headers)
            if grp_id not in self.fitter_table:
                self.fitter_table[grp_id] = {
                    "train_x": [], "train_y": [],
                    "test_x": [], "test_y": []}
            self.fitter_table[grp_id]["train_x"].append(train_x[i])
            self.fitter_table[grp_id]["train_y"].append(train_y[i])

        for i in range(test_x.shape[0]):
            grp_id = self.gen_grp_id(dels, test_x[i], test_y[i], headers)
            if grp_id not in self.fitter_table:
                self.fitter_table[grp_id] = {
                    "train_x": [], "train_y": [],
                    "test_x": [], "test_y": []
                }
            self.fitter_table[grp_id]["test_x"].append(test_x[i])
            self.fitter_table[grp_id]["test_y"].append(test_y[i])

    def gen_grp_id(self, delimiters, xdata, ydata, headers):
        """
        Parameters
        ----------
        delimiters: a list of class `Delimiter`, containing traget dimension, fd_len, ... used for division
        xdata : shape = (n_features)
        ...
        """
        grp_ids = []
        for delimiter in delimiters:
            ### select the refer according to the target dimension
            if delimiter.target_dim == 'avg':
                x = ydata[0]
            elif delimiter.target_dim == 'intensity':
                raise
            else:
                ### target dimension index needs to exclude avg
                target_idx = headers.index(delimiter.target_dim) - 1
                x = xdata[target_idx]
            grp_id = "%03d_%03d"%(int(x / delimiter.td_len), int((x % delimiter.td_len) % delimiter.fd_len / delimiter.unit_len)) \
                if delimiter.fd_len > 0 else "%03d"%(int(x / delimiter.td_len))
            grp_ids.append(grp_id)
        return '-'.join(grp_ids)

    def print_dels(self, dels):
        rst = None
        for _del in dels:
            if rst is None: 
                rst = "del={}, td_len={}, fd_len={}, unit_len={}, max_grp_size={}".format(_del.target_dim, _del.td_len, _del.fd_len, _del.unit_len, _del.max_grp_size)
            else:
                rst += "\ndel={}, td_len={}, fd_len={}, unit_len={}, max_grp_size={}".format(_del.target_dim, _del.td_len, _del.fd_len, _del.unit_len, _del.max_grp_size)
        return rst

    def train_test(self, dels, headers, op_type="conv", visual=True):
        if not isinstance(dels, list):
            dels = [dels]
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
            try:
                popt, pcov = self.fitter_table[grp_id]["fitter"].train()
                error = self.fitter_table[grp_id]["fitter"].test(verbose=True)
            except RuntimeError:
                print("[WARNING] RuntimeError")
                error = None

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
            ax.set_xlabel('Group ID')
            ax.bar(xaxis, plots[1], label='training data size', width=bar_width, color=clrs[0])
            ax.bar(xaxis + bar_width, plots[2], label='test data size', width=bar_width, color=clrs[2])
            plt.xticks(xaxis + bar_width/2, plots[0], rotation=80)
            plt.legend(loc=2)
            
            ax = ax.twinx()
            ax.plot(plots[0], plots[3], '.-', label='fitting error = %6.4f %%'%(sum(plots[3])/len(plots[3])))
            ax.set_ylabel("Fitting error (%)")
            plt.legend(loc=1)

            ax.tick_params(axis='y', labelcolor=clrs[3])
            plt.title("Data size and fitting error of each group \n{}".format(self.print_dels(dels)))

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()


            






