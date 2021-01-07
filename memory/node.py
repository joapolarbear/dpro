import numpy as np


class Node:
    def __init__(self, name, op, input, dtype, shape):
        self._name = name
        self._op = op
        self._input = input
        self._dtype = dtype
        self._shape = shape
        self._requires_grad = True
        self._inplace = False

    @classmethod
    def from_metadata(cls, name, metadata):
        """create node from metadata

        Args:
            name (str): node name
            metadata (dict): from metadata.json

        Returns:
            [Node]: created node
        """
        def _get_op(node_def):
            return node_def.get("op").lower()

        def _get_input(node_def):
            inputs = node_def.get("input")
            if not inputs:
                return None

            names = []
            for input in inputs:
                full_name = input.get("name")
                name = full_name.rsplit(':')[0]
                names.append(name)
            return names

        def _get_dtype(node_def):
            output = node_def.get("output")
            if not output:
                return None
            dtype = output[0].get("dtype")
            if not dtype:
                return None
            return np.dtype(dtype)

        def _get_shape(node_def):
            output = node_def.get("output")
            if not output:
                return None
            return tuple(output[0].get("shape"))

        if name not in metadata:
            return None

        node_def = metadata[name]

        return cls(name,
                   _get_op(node_def),
                   _get_input(node_def),
                   _get_dtype(node_def),
                   _get_shape(node_def))

    def is_valid(self):
        """check the node's validity 

        Returns:
            [bool]: validity
        """
        def _is_valid_op(op):
            if op in ["NoOp"]:
                return False
            return True

        def _is_not_none(value):
            if value is None:
                return False
            return True

        def _is_valid_shape(shape):
            if not isinstance(shape, tuple):
                return False
            if not shape or shape[0] == -1:
                return False
            return True

        return all([
            _is_valid_op(self.op),
            _is_not_none(self.dtype),
            _is_not_none(self.input),
            _is_valid_shape(self.shape)
        ])

    def is_parameter(self):
        """Whether this node is parameter node

        Returns:
            [bool]: is parameter node
        """
        if self._op == "variablev2":
            return True
        return False
    
    def get_num_ele(self):
        """get number of elements

        Returns:
            [int]: number of elements
        """
        return np.prod(self.shape)

    def get_output_size(self):
        """get output size

        Returns:
            [float]: size in Byte
        """
        if self._inplace:
            return 0
        return np.prod(self.shape) * self.dtype.itemsize

    def get_temp_size(self):
        """get temporary buffer size

        useful for cudnn workspace size

        Returns:
            [foat]: size in Byte
        """
        return 0

    @property
    def name(self):
        """get name

        Returns:
            [str]: node name
        """
        return self._name

    @property
    def op(self):
        """get operator type

        Returns:
            [str]: operator type
        """
        return self._op

    @property
    def input(self):
        """get input list

        Returns:
            [list]: input node name list
        """
        return self._input

    @property
    def dtype(self):
        """get data type 

        Returns:
            [numpy.dtype]: data type
        """
        return self._dtype

    @property
    def shape(self):
        """get output shape

        Returns:
            [tuple]: output shape
        """
        return self._shape

    @property
    def requires_grad(self):
        """get requires_grad

        Returns:
            [bool]: requires_grad
        """
        return self._requires_grad

    @property
    def inplace(self):
        """get inplace status

        Returns:
            [bool]: inplace
        """
        return self._inplace

    @inplace.setter
    def inplace(self, val):
        self._inplace = val

    def __repr__(self):
        return "Name: %s, op: %s, input: [%s], dtype: %s, shape: %s" % (
            self.name, self.op, ", ".join(self.input), str(
                self.dtype), str(self.shape)
        )
