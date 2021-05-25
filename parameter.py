''' Manage the parameter info of a DNN model
'''
import re
from trace_utils import *

class ParameterDict:
    def __init__(self, _pm, platform, metadata_path=None):
        ### collect metadata
        if metadata_path is None:
            metadata_path = os.path.dirname(_pm.search(FileName.METADATA))
        if metadata_path is None:
            SingleLogger().error(
                "{} not found. Fail to load metadata".format(FileName.METADATA.value))
        
        if platform == "MXNET":
            from ml_platform.mxnet.metadata import MetaInfo
            SingleLogger().info("Use MXNET metadata")
        elif platform == "TENSORFLOW":
            from ml_platform.tensorflow.metadata import MetaInfo
            SingleLogger().info("Use TENSORFLOW metadata")
        else:
            raise NotImplementedError()

        self.metainfo = MetaInfo(metadata_path)
        self.cnt = len(self.metainfo.gradient_name_list)

    def gradient_name_list(self):
        return self.metainfo.gradient_name_list
    
    def gradient_num(self):
        return self.cnt

    def wrap_read_dfg(self,  *args, **kwargs):
        return self.metainfo.wrap_read_dfg(*args, **kwargs)
    
    def standard_name(self,  op_name):
        ''' Convert op_names in the original traces to standard names
            `op_cat.op_name.sub_op`
        '''
        return self.metainfo.standard_name(op_name)
    
    ### below methods are related to tensors/Communication

    def tensor_id_to_tensor_name(self, tensor_id):
        return self.metainfo.tensor_id_to_tensor_name(tensor_id)

    def tensor_name_to_tensor_id(self, name):
        return self.metainfo.tensor_name_to_tensor_id(name)

    def tensor_id2size(self, tensor_id):
        return self.metainfo.ret_tensor_size(tensor_id)
    
    def tensor_id2update_id(self, tensor_id):
        '''tensor id may be 'max' to return the maximum update id '''
        return self.metainfo.tensor_id2update_id(tensor_id)
    
    ### below is related op_name

    def ret_metadata(self, *args, **kwargs):
        return self.metainfo.ret_metadata(*args, **kwargs)

    def ret_rawmeta(self, op_name):
        return self.metainfo.ret_rawmeta(op_name)
    
    def check_amp_lists(self, op_name):
        return self.metainfo.check_amp_lists(op_name)
    
    def parse_op_type(self, op_name):
        return self.metainfo.parse_op_type(op_name)
    
    def ret_op_precision(self, op_name):
        return self.metainfo.ret_op_precision(op_name)
    
    def in_metadata(self, op_name):
        return self.metainfo.in_metadata(op_name)
    
    def is_const(self, op_name):
        return self.metainfo.is_const(op_name)
    
    def is_variable(self, op_name):
        return self.metainfo.is_variable(op_name)
    
