''' This module is used to filter operators that we focus on for AMP
    And write them in avg.txt file
    write corresponding names in name.txt
'''
import sys, os

class TraceFilter:
    def __init__(self, save_names=None, model=None, platform=None):
        self.save_names = save_names
        self.platform = platform
        self.model = model

        if self.platform == 'tf':
            MNIST_CANDIDATES = ["Conv2D", "BiasAdd", "Relu", "MatMul", "Mul", "Cast", "BiasAddGrad", "ApplyAdam", "ReluGrad", "Conv2DBackpropInput", "Conv2DBackpropFilter"]
            RESNET50_CANDIDATES = ["Conv2D", "BiasAdd", "Relu", "MatMul", "Mul", "Cast"]
        else:
            MNIST_CANDIDATES = RESNET50_CANDIDATES = ["conv", "BiasAdd", "Relu", "MatMul", "Mul", "Cast"]

        if "resnet" in self.model.lower():
            self._CANDIDATES = RESNET50_CANDIDATES
        elif "mnist" in self.model.lower():
            self._CANDIDATES = MNIST_CANDIDATES
        elif "bert" in self.model.lower():
            self._CANDIDATES = None
        elif "dense" in self.model.lower():
            self._CANDIDATES = ["_dense", "MatMul", "Mat", "Cast"]
        else:
            self._CANDIDATES = None

    def _is_ignore_for_sta(self, name):
        ### store the pid for computation
        if self._CANDIDATES is None:
            return False
        for target in self._CANDIDATES:
            if target in name:
                return False
        return True

    def dump_for_cost_model(self, name2sta, _dir):
        nameL = []
        avg = []
        var = []
        for name, statistic in sorted(name2sta.items()):
            if self._is_ignore_for_sta(name):
                continue    
            nameL.append(name)
            avg.append(statistic["avg"])
            var.append(statistic["var"])
        # print(nameL, avg)
        if self.save_names != "None":
            with open(os.path.join(_dir, "name.txt"), "a") as f:
                f.write("{}:{}\n".format(self.save_names, str(nameL)))
        with open(os.path.join(_dir, "avg.txt"), "a") as f:
            f.write(str(avg) + "\n")
            f.write(str(var) + "\n")

   






