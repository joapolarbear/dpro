import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pickle
import os
import scipy.interpolate as interpolate

from .gpu_cost_model import CurveFiter
from ...logger_utils import SingleLogger

class Delimiter:
    def __init__(self, target_dim, td_len=1., fd_len=0., unit_len=1., max_grp_size=20):
        """
        Parameters
        ----------
        target_dim : str, the target dimension used to divide the dataset
        td_len : integer
        """
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
    def __init__(self, dels=None, headers=None, op_type=None, max_of_each_dim=None):
        '''
        Parameters
        ----------
        headers: list of str
            List of dimension names
        max_of_each_dim: list of float
            Maximum value of each dimension, n_dum == len(headers)
        '''
        self.dels = dels
        self.headers = headers
        self.max_of_each_dim = max_of_each_dim
        self.op_type = op_type

        self.fitter_table = {}

    def divide_with_upper(self, train_x, train_y, test_x, test_y, visual=True):
        if self.dels.target_dim != 'avg' and self.dels.target_dim != 'intensity':
            target_idx = self.headers.index(self.dels.target_dim) - 1

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
            if grp_size >= self.dels.max_grp_size:
                grp_id += 1
                grp_size = 0

    def divide_by_len(self, train_x, train_y, test_x, test_y):
        """
        Parameters
        ----------
        dels: class `Delimiter` or a list of class `Delimiter`, containing traget dimension, fd_len, ... used for division
        train_x : shape = (n_samples, n_features)
        ...
        """
        if not isinstance(self.dels, list):
            self.dels = [self.dels]

        for i in range(train_x.shape[0]):
            grp_id = self.gen_grp_id(train_x[i], train_y[i])
            if grp_id not in self.fitter_table:
                self.fitter_table[grp_id] = {
                    "train_x": [], "train_y": [],
                    "test_x": [], "test_y": []}
            self.fitter_table[grp_id]["train_x"].append(train_x[i])
            self.fitter_table[grp_id]["train_y"].append(train_y[i])

        for i in range(test_x.shape[0]):
            grp_id = self.gen_grp_id(test_x[i], test_y[i])
            if grp_id not in self.fitter_table:
                self.fitter_table[grp_id] = {
                    "train_x": [], "train_y": [],
                    "test_x": [], "test_y": []
                }
            self.fitter_table[grp_id]["test_x"].append(test_x[i])
            self.fitter_table[grp_id]["test_y"].append(test_y[i])
        
        for grp_id in sorted(self.fitter_table.keys()):
            self.fitter_table[grp_id]["train_x"] = np.array(self.fitter_table[grp_id]["train_x"])
            self.fitter_table[grp_id]["train_y"] = np.array(self.fitter_table[grp_id]["train_y"])
            self.fitter_table[grp_id]["test_x"] = np.array(self.fitter_table[grp_id]["test_x"])
            self.fitter_table[grp_id]["test_y"] = np.array(self.fitter_table[grp_id]["test_y"])
        SingleLogger().info("Total number of groups: {}".format(len(self.fitter_table)))

    def gen_grp_id(self, xdata, ydata):
        """
        Parameters
        ----------
        self.dels: a list of class `Delimiter`, containing traget dimension, fd_len, ... used for division
        xdata : shape = (n_features)
        ...
        """
        grp_ids = []
        for delimiter in self.dels:
            ### select the refer according to the target dimension
            if delimiter.target_dim == 'avg':
                x = ydata[0]
            elif delimiter.target_dim == 'intensity':
                x = xdata[self.headers.index('S_mul') - 1] / (xdata[self.headers.index('S_in') - 1] + xdata[self.headers.index('S_out') - 1] + xdata[self.headers.index('S_wei') - 1])
            else:
                ### target dimension index needs to exclude avg
                target_idx = self.headers.index(delimiter.target_dim) - 1
                x = xdata[target_idx]
            grp_id = "%03d_%03d"%(int(x / delimiter.td_len), int((x % delimiter.td_len) % delimiter.fd_len / delimiter.unit_len)) \
                if delimiter.fd_len > 0 else "%03d"%(int(x / delimiter.td_len))
            grp_ids.append(grp_id)
        if len(grp_ids) == 0:
            return "default"
        return '-'.join(grp_ids)

    def print_dels(self, dels):
        rst = None
        for _del in dels:
            if rst is None: 
                rst = "del={}, td_len={}, fd_len={}, unit_len={}, max_grp_size={}".format(_del.target_dim, _del.td_len, _del.fd_len, _del.unit_len, _del.max_grp_size)
            else:
                rst += "\ndel={}, td_len={}, fd_len={}, unit_len={}, max_grp_size={}".format(_del.target_dim, _del.td_len, _del.fd_len, _del.unit_len, _del.max_grp_size)
        return rst

    def train_all(self):        
        for grp_id in sorted(self.fitter_table.keys()):
            self.fitter_table[grp_id]["fitter"] = CurveFiter(self.headers, op_type=self.op_type)
            SingleLogger().info("Group ID {} collects training data - X:{}, Y:{}".format(
                    grp_id,
                    self.fitter_table[grp_id]["train_x"].shape, 
                    self.fitter_table[grp_id]["train_y"].shape))
            try:
                popt, pcov = self.fitter_table[grp_id]["fitter"].train(
                    self.fitter_table[grp_id]["train_x"],
                    self.fitter_table[grp_id]["train_y"])
            except RuntimeError:
                SingleLogger().warn("[WARNING] RuntimeError")

    def test_all(self, visual=True, dump_path=None):
        if not isinstance(self.dels, list):
            self.dels = [self.dels]
        plots = [[], [], [], []]
        for grp_id in sorted(self.fitter_table.keys()):
            SingleLogger().info("Group ID {} collects test data - X:{}, Y:{}".format(
                grp_id,
                self.fitter_table[grp_id]["test_x"].shape,
                self.fitter_table[grp_id]["test_y"].shape))
            error = self.fitter_table[grp_id]["fitter"].test(
                self.fitter_table[grp_id]["test_x"],
                self.fitter_table[grp_id]["test_y"],
                verbose=True)
            if error is not None:
                plots[0].append(grp_id)
                plots[1].append(self.fitter_table[grp_id]["train_x"].shape[0])
                plots[2].append(self.fitter_table[grp_id]["test_x"].shape[0])
                plots[3].append(error)

        fitting_error = sum(plots[3])/len(plots[3])
        if visual:
            plt.figure(num=1, figsize=(8, 4))
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
            ax.plot(plots[0], plots[3], '.-', label='fitting error = %6.4f %%'%(fitting_error))
            ax.set_ylabel("Fitting error (%)")
            plt.legend(loc=1)

            ax.tick_params(axis='y', labelcolor=clrs[3])
            plt.title("Data size and fitting error of each group \n{}".format(self.print_dels(self.dels)))

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            if dump_path is None:
                plt.show()
            else:
                plt.savefig(dump_path)
        return fitting_error

    def predict(self, xdata, normalized=False):
        ### normalized 
        if not normalized:
            xdata = xdata / self.max_of_each_dim[1:]
        if len(xdata.shape) == 1:
            return self._predict(xdata)
        else:
            rst_list = []
            for i in range(xdata.shape[0]):      
                rst_list.append(self._predict(xdata[i]))
            return np.array(rst_list) * self.max_of_each_dim[0]

    def _predict(self, _xdata):
        grp_id = self.gen_grp_id(_xdata, None)
        if grp_id not in self.fitter_table:
            rst = self._interpolate_predict(_xdata, grp_id)
        else:
            rst = self.fitter_table[grp_id]["fitter"].predict(_xdata)
        rst = rst * self.max_of_each_dim[0]
        # print("MP cost model: {}(group {}) predicts {} ms".format(self.op_type, grp_id, rst))
        return rst

    def _grp_id2list(self, grp_id):
        return [int(gid) for gid in grp_id.split("-")]

    def _interpolate_predict(self, _xdata, grp_id):
        grp_id_list = np.array(self._grp_id2list(grp_id))
        ### shape = (n_grps, n_dels)
        all_grp_id_list = np.array([self._grp_id2list(gid)
                                    for gid in self.fitter_table.keys()])
        ### shape = (n_grps, n_popts)
        popt_list = np.array([_dict["fitter"].popt
                                    for _dict in self.fitter_table.values()])
        ret_popt = interpolate.LinearNDInterpolator(all_grp_id_list, popt_list)(grp_id_list)
        if np.isnan(np.min(ret_popt)):
            ret_popt = interpolate.NearestNDInterpolator(all_grp_id_list, popt_list)(grp_id_list)
        self.fitter_table[grp_id]["fitter"] = CurveFiter(self.headers, op_type=self.op_type)
        self.fitter_table[grp_id]["fitter"].popt = ret_popt
        return self.fitter_table[grp_id]["fitter"].predict(_xdata)

    def dump(self):
        cost_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cost_model")
        if not os.path.exists(cost_model_dir):
            os.mkdir(cost_model_dir)
        
        ### function member can not be dumped with pickler
        for grp_id, dict_ in self.fitter_table.items():
            dict_["fitter"].fit_func = None
        
        with open(os.path.join(cost_model_dir, "{}.txt".format(self.op_type)), "wb") as f:
            pickle.dump([self.dels, self.headers, self.max_of_each_dim,
                         self.op_type, self.fitter_table], f)

    def load(self, cm_path):
        if not os.path.exists(cm_path):
            SingleLogger().error("No AMP cost model at {}".format(cm_path))
        with open(cm_path, "rb") as f:
            self.dels, self.headers, self.max_of_each_dim, self.op_type, \
                self.fitter_table = pickle.load(f)
        
        SingleLogger().info("Load AMP cost model for {} ...".format(self.op_type))
        for grp_id in self.fitter_table:
            SingleLogger().info(" - Load group {}".format(grp_id))
            self.fitter_table[grp_id]["fitter"].load_fit_func()

def load_grouper(cm_path):
    grp = Grouper()
    grp.load(cm_path)
    return grp

 


