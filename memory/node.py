import numpy as np
import tensorflow as tf


class Node:
    def __init__(self, node_def):
        self._name = self._get_name(node_def)
        self._op = self._get_op(node_def)
        self._input = self._get_input(node_def)
        self._dtype = self._get_dtype(node_def)
        self._shape = self._get_shape(node_def)

    @staticmethod
    def _get_name(node_def):
        return node_def["name"]

    @staticmethod
    def _get_op(node_def):
        return node_def["op"].lower()

    @staticmethod
    def _get_input(node_def):
        if "input" not in node_def:
            return None
        return node_def["input"]

    @staticmethod
    def _get_dtype(node_def):
        def _convert_tf_dtype_to_np(tf_dtype):
            tf_dtype = tf.dtypes.DType(tf_dtype)
            return np.dtype(tf_dtype.name)

        if "dtype" not in node_def:
            return None
        tf_dtype = node_def["dtype"]
        if tf_dtype:
            return _convert_tf_dtype_to_np(tf_dtype)

        return None

    @staticmethod
    def _get_shape(node_def):
        if "shape" not in node_def:
            return None

        shape = node_def["shape"]
        # scalar
        if not shape:
            shape = [1, ]

        return tuple(shape)

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

    def __repr__(self):
        return "Name: %s, op: %s, input: [%s], dtype: %s, shape: %s" % (
            self.name, self.op, ", ".join(self.input), str(
                self.dtype), str(self.shape)
        )
