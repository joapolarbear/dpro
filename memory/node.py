import numpy as np 

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class Node:
    def __init__(self, node_def):
        self._name = Node._get_name(node_def)
        self._op = Node._get_op(node_def)
        self._input = Node._get_input(node_def)
        self._dtype = Node._get_dtype(node_def)
        self._shape = Node._get_shape(node_def)

    @staticmethod
    def _get_name(node_def):
        return node_def["name"]

    @staticmethod
    def _get_op(node_def):
        return node_def["op"]

    @staticmethod
    def _get_input(node_def):
        return node_def["input"]

    @staticmethod
    def _get_dtype(node_def):
        attr = node_def["attr"]
        tf_dtype = Node._get_tf_dtype(attr)
        if tf_dtype:
            return Node._convert_tf_dtype_to_np(tf_dtype)            

        return None

    @staticmethod
    def _get_shape(node_def):
        pass

    @staticmethod
    def _get_tf_dtype(attr):
        if "dtype" in attr:
            return attr["dtype"]["type"]
        elif "T" in attr:
            return attr["T"]["type"]
        else:
            return None

    @staticmethod
    def _convert_tf_dtype_to_np(tf_dtype):
        dtype = remove_prefix(tf_dtype, "DT_")
        dtype = dtype.lower()
        
        # "float" denotes "double" in numpy
        if dtype == "float":
            dtype = "float32"
        elif dtype == "string":
            dtype = "str"
        
        return np.dtype(dtype)

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
