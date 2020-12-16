from cost_model_xla.gen_samples import GSNotInGraphError
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras

from tensorflow.keras import layers, Model
import shutil
import copy
import random
from multiprocessing import Pool
from tqdm import tqdm

import scipy.optimize as scipy_optimize

import os
import pickle
try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef
from cost_model_xla.xlatools import compile_to_hlo, extract_kernel_features_from_hlo, replay_and_generate_kernel_sample, BPF_PROFILE_GPU
from google.protobuf.json_format import Parse
from cost_model_xla.gen_dataset_utils import XlaKernelDataset, XlaModuleTestSet
from cost_model_xla.gen_samples import GSNotInGraphError, GSNonFixedShapeError, GSSubgraphTooSmallError, GraphDefUtil
from tqdm import tqdm, trange
from collections import defaultdict
import traceback

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to use other GPUS
  avoided_gpu_id = BPF_PROFILE_GPU
  for dev in gpus:
      dev_id = int(dev.name.split("GPU:")[-1])
      if dev_id != avoided_gpu_id:
        tf.config.experimental.set_visible_devices(dev, 'GPU')
        print("Set cost model to use GPU {}".format(dev_id))
        break

class ElementaryOpCache():
    def __init__(self, dataset_path=None, load_from=None):
        if load_from is not None:
            self.load(load_from)
        else:
            if dataset_path is None:
                raise RuntimeError("At least one of dataset_path and load_from must be set.")
            self._dataset_path = dataset_path
            self._cache_path = os.path.join(dataset_path, "elementary_ops.txt")
            self._load_cache()
    
    def _load_cache(self):
        total_counter = 0
        hash_dict = defaultdict(list)
        op_code_dict = defaultdict(list)
        with open(self._cache_path, "r") as f:
            for line in f:
                hash_value, op_code, time = [v.strip() for v in line.split(",")]
                hash_value = int(hash_value)
                time = float(time)
                hash_dict[hash_value].append(time)
                op_code_dict[op_code].append(time)
                total_counter += 1
        self.hash_dict = hash_dict
        self.op_code_dict = op_code_dict
        psd = []
        lens = []
        counter = 0
        for key, times in hash_dict.items():
            lens.append(len(times))
            if np.max(times) != 0:
                counter += 1
            if len(times) >= 1 and np.average(times) != 0:
                psd.append(np.std(times) / np.average(times))

        print("[OP Cache INFO] Total recorded elementary ops: {}, deduplicated size of elementary ops: {}, non_zero ops: {}".format(total_counter ,len(hash_dict), counter))
        print("[OP Cache INFO] Mean PSD: {}, Max PSD: {}, Min PSD: {}, \n max collected len: {}, min collected len: {}, mean collected len: {}".format(np.average(psd), np.max(psd), np.min(psd), np.max(lens), np.min(lens), np.average(lens)))

        elemophash2index = {}
        index2elemophash = {}
        sorted_elem_op_hashes = sorted(list(self.hash_dict.keys()))
        for index, op_hash in enumerate(sorted_elem_op_hashes):
            elemophash2index[op_hash] = index
            index2elemophash[index] = op_hash
        
        self.elemophash2index = elemophash2index
        self.index2elemophash = index2elemophash
    
    def query(self, hash_v, op_code=None):
        if hash_v not in self.hash_dict:
            if op_code is not None and op_code in self.op_code_dict:
                return np.average(self.op_code_dict[op_code]), False
            else:
                return 0, False
        else:
            return np.average(self.hash_dict[hash_v]), True

    def dump(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump([self._dataset_path, self._cache_path, 
                        self.hash_dict, self.op_code_dict,
                        self.elemophash2index, self.index2elemophash], f)
    
    def load(self, save_path):
        with open(save_path, "rb") as f:
            (self._dataset_path, self._cache_path, 
            self.hash_dict, self.op_code_dict,
            self.elemophash2index, self.index2elemophash) = pickle.load(f)

class XLAModuleOverheadModel():
    def __init__(self, dataset_path = None, dataset=None, elem_op_cache=None, load_from=None, use_dual_model=True, split_threshold=10, regularization_lambda = 0.1):
        if load_from is not None:
            self.load(load_from)
        else:
            self.regularization_lambda = regularization_lambda
            self.use_dual_model = use_dual_model
            self.split_threshold = split_threshold
            if dataset_path is None or dataset is None or elem_op_cache is None:
                raise RuntimeError("Must set all parameters if not loading from cache.")
            self.dataset = dataset
            self.elem_op_cache = elem_op_cache
            self._dataset_path = dataset_path
            self._read_execution_times()
            self._preprocess_data()
    
    def _process_config_and_exec_file(self, sample_id):
        config_path = os.path.join(self._modules_dir_path, str(sample_id) + "_config.txt")
        sample_details = []
        sample_times = None
        subop_exec_time = 0
        max_dim = 0
        abnormal = False
        with open(config_path, "r") as f:
            for line in f:
                is_fused_op = bool(int(line.split(",")[0]))
                if is_fused_op:
                    fused_op_hash = int(line.split(",")[1])
                    kernel_path = os.path.join(self._features_dir_path, line.split(",")[-1].split("/")[-1].strip())
                    op_count = -1
                    with open(kernel_path, "r") as f_kernel:
                        for kernel_line in f_kernel:
                            op_count += 1
                    kernel_sid = int(line.split(",")[-1].split("/")[-1].split(".txt")[0])
                    # get fusion execution time
                    if kernel_sid in self.dataset.label_dict:
                        exec_time = self.dataset.label_dict[kernel_sid]
                    else:
                        exec_time = self.dataset.label_dict[self.dataset.dedupedsid2finalsid[kernel_sid]]
                    sample_details.append(len(self.elem_op_cache.index2elemophash) -1 + op_count)
                    max_dim = max(max_dim, len(self.elem_op_cache.index2elemophash) + op_count)
                else:
                    _, elem_op_hash, op_code = [v.strip() for v in line.split(",")]
                    elem_op_hash = int(elem_op_hash)
                    exec_time, _ = self.elem_op_cache.query(elem_op_hash, op_code=op_code)
                    sample_details.append(self.elem_op_cache.elemophash2index[elem_op_hash])
                subop_exec_time += exec_time
        # read exec.txt
        exec_path = os.path.join(self._modules_dir_path, str(sample_id) + "_exec.txt")
        exec_times = []
        with open(exec_path, "r") as f: 
            for line in f:
                exec_times.append(float(line.strip()))
        if len(exec_times) > 100:
            module_avg_time = np.average(exec_times[-100:-10])
        else:
            module_avg_time = np.average(exec_times[10:-10])
        if module_avg_time > subop_exec_time:
            sample_times = (module_avg_time, subop_exec_time)
        else:
            abnormal = True
        return sample_id, sample_details, sample_times, abnormal, max_dim

    def _read_execution_times(self):
        self._modules_dir_path = os.path.join(self._dataset_path, "modules")
        self._features_dir_path = os.path.join(self._dataset_path, "features")
        sample_id_set = set()
        for fn in os.listdir(self._modules_dir_path):
            sample_id = int(fn.split(".txt")[0].split("_")[0])
            sample_id_set.add(sample_id)

        # def add_config_suffix(sid):
        #     return str(sid) + "_config.txt"

        # def add_exec_suffix(sid):
        #     return str(sid) + "_exec.txt"

        module_time_dict = {}
        module_details_dict = {}
        max_dim = len(self.elem_op_cache.index2elemophash)
        abnormal_count = 0

        num_cores = min(os.cpu_count(), len(sample_id_set))
        chunk_size = int( np.ceil(len(sample_id_set) / num_cores / 2))

        with Pool(num_cores) as p:
            sample_details = list(tqdm(p.imap_unordered(self._process_config_and_exec_file, sample_id_set, chunksize=chunk_size), total=len(sample_id_set), desc="Reading details", leave=False))
        
        for (sample_id, sample_details, sample_times, abnormal, max_dim_in_sample) in sample_details:
            if sample_id not in module_details_dict:
                module_details_dict[sample_id] = []
            max_dim = max(max_dim, max_dim_in_sample)
            if abnormal:
                abnormal_count += 1
                continue
            module_details_dict[sample_id] = sample_details
            module_time_dict[sample_id] = sample_times

        # for sample_id in tqdm(sample_id_set, desc="Reading details:"):
        #     if sample_id not in module_details_dict:
        #         module_details_dict[sample_id] = []
        #     # read config.txt
        #     config_path = os.path.join(self._modules_dir_path, add_config_suffix(sample_id))
        #     subop_exec_time = 0
        #     with open(config_path, "r") as f:
        #         for line in f:
        #             is_fused_op = bool(int(line.split(",")[0]))
        #             if is_fused_op:
        #                 fused_op_hash = int(line.split(",")[1])
        #                 kernel_path = os.path.join(self._features_dir_path, line.split(",")[-1].split("/")[-1].strip())
        #                 op_count = -1
        #                 with open(kernel_path, "r") as f_kernel:
        #                     for kernel_line in f_kernel:
        #                         op_count += 1
        #                 kernel_sid = int(line.split(",")[-1].split("/")[-1].split(".txt")[0])
        #                 # get fusion execution time
        #                 if kernel_sid in self.dataset.label_dict:
        #                     exec_time = self.dataset.label_dict[kernel_sid]
        #                 else:
        #                     exec_time = self.dataset.label_dict[self.dataset.dedupedsid2finalsid[kernel_sid]]
        #                 module_details_dict[sample_id].append(len(self.elem_op_cache.index2elemophash) -1 + op_count)
        #                 max_dim = max(max_dim, len(self.elem_op_cache.index2elemophash) + op_count)
        #             else:
        #                 _, elem_op_hash, op_code = [v.strip() for v in line.split(",")]
        #                 elem_op_hash = int(elem_op_hash)
        #                 exec_time, _ = self.elem_op_cache.query(elem_op_hash, op_code=op_code)
        #                 module_details_dict[sample_id].append(self.elem_op_cache.elemophash2index[elem_op_hash])
        #             subop_exec_time += exec_time
        #     # read exec.txt
        #     exec_path = os.path.join(self._modules_dir_path, add_exec_suffix(sample_id))
        #     exec_times = []
        #     with open(exec_path, "r") as f: 
        #         for line in f:
        #             exec_times.append(float(line.strip()))
        #     if len(exec_times) > 100:
        #         module_avg_time = np.average(exec_times[-100:-10])
        #     else:
        #         module_avg_time = np.average(exec_times[10:-10])
        #     if module_avg_time > subop_exec_time:
        #         module_time_dict[sample_id] = (module_avg_time, subop_exec_time)
        #     else:
        #         abnormal_count += 1
        #         module_details_dict.pop(sample_id)
        print("Processed {} samples, with {} have module time <= subop exec time.".format(len(sample_id_set), abnormal_count))
        
        self.module_details_dict = module_details_dict
        self.module_time_dict = module_time_dict
        self.max_dim = max_dim
        self.fusion_op_offset = len(self.elem_op_cache.index2elemophash) - 1
    
    def _preprocess_data(self):
        # generate the matrix A, and vector b
        self.row_dim = len(self.module_details_dict)
        self.column_dim = self.max_dim + 1
        A = np.zeros((self.row_dim, self.column_dim))
        b = np.zeros((self.row_dim,))
        for row_index, key in enumerate(self.module_details_dict.keys()):
            # fill A
            A[row_index, 0] = 1
            for column_index in self.module_details_dict[key]:
                A[row_index, column_index + 1] += 1
            # fill B
            module_avg_time, subop_exec_time = self.module_time_dict[key]
            b[row_index] = module_avg_time - subop_exec_time

        A_normed = A.copy()
        A_normed = (A_normed.T / b).T

        if self.use_dual_model:
            small_mask = np.sum(A, axis=1) < self.split_threshold
            A_normed_small = A_normed[small_mask]
            b_normed_small = np.ones((A_normed_small.shape[0], ))
            A_normed_large = A_normed[~small_mask]
            b_normed_large = np.ones((A_normed_large.shape[0], ))
        else:
            A_normed_small = None
            b_normed_small = None
            A_normed_large = None
            b_normed_large = None

        self.A = A
        self.A_normed = A_normed
        self.b = b
        self.b_normed = np.ones((self.row_dim, ))
        self.A_normed_small = A_normed_small
        self.b_normed_small = b_normed_small
        self.A_normed_large = A_normed_large
        self.b_normed_large = b_normed_large

    def fit(self):
        if self.use_dual_model:
            reg_A = self.regularization_lambda * np.eye(self.A_normed_large.shape[1])
            reg_b = np.zeros(self.A_normed_large.shape[1])
            coeffs = None
            residual = None
            coeffs_large, residual_large = scipy_optimize.nnls(np.concatenate((self.A_normed_large, reg_A), axis=0), np.concatenate((self.b_normed_large, reg_b), axis=0))
            coeffs_small, residual_small = scipy_optimize.nnls(np.concatenate((self.A_normed_small, reg_A), axis=0), np.concatenate((self.b_normed_small, reg_b), axis=0))
            self.avg_elem_ovhd_large = np.average(coeffs_large[1:-1])
            self.avg_elem_ovhd_small = np.average(coeffs_small[1:-1])
            self.avg_elem_ovhd = None
        else:
            reg_A = self.regularization_lambda * np.eye(self.A_normed.shape[1])
            reg_b = np.zeros(self.A_normed.shape[1])
            coeffs, residual = scipy_optimize.nnls(np.concatenate((self.A_normed, reg_A), axis=0), np.concatenate((self.b_normed, reg_b), axis=0))
            self.avg_elem_ovhd_large = None
            self.avg_elem_ovhd_small = None
            self.avg_elem_ovhd = np.average(coeffs[1:-1])
            coeffs_large = residual_large = coeffs_small = residual_small = None
        # print("Fitted model. Residual: {}".format(residual))
        self.coeffs = coeffs
        self.coeffs_large = coeffs_large
        self.coeffs_small = coeffs_small

    def refit(self, use_dual_model=True, split_threshold=5, regularization_lambda = 0.1):
        self.use_dual_model = use_dual_model
        self.split_threshold = split_threshold
        self.regularization_lambda = regularization_lambda
        self._preprocess_data()
        self.fit()
    
    def reinit(self, use_dual_model=True, split_threshold=5, regularization_lambda = 0.1):
        self._read_execution_times()
        self.refit(use_dual_model, split_threshold, regularization_lambda)
    
    # def get_overhead(self, elem_op_hashes, num_fused_ops):
    def get_overhead(self, elem_op_hashes, subop_lengths):
        sum_ovhds = 0
        num_ops = len(elem_op_hashes) + len(subop_lengths)
        for (hash_v, op_code) in elem_op_hashes:
            if hash_v in self.elem_op_cache.elemophash2index:
                if self.use_dual_model:
                    if num_ops < self.split_threshold:
                        # use small model
                        sum_ovhds += self.coeffs_small[self.elem_op_cache.elemophash2index[hash_v]]
                    else:
                        sum_ovhds += self.coeffs_large[self.elem_op_cache.elemophash2index[hash_v]]
                else:
                    sum_ovhds += self.coeffs[self.elem_op_cache.elemophash2index[hash_v]]
            else:
                if self.use_dual_model:
                    if num_ops < self.split_threshold:
                        # use small model
                        sum_ovhds += self.avg_elem_ovhd_small
                    else:
                        sum_ovhds += self.avg_elem_ovhd_large
                else:
                    sum_ovhds += self.avg_elem_ovhd
        if self.use_dual_model:
            if num_ops < self.split_threshold:
                # use small model
                sum_ovhds += self.coeffs_small[0]
                # sum_ovhds += self.coeffs_small[-1] * num_fused_ops
                for length in subop_lengths:
                    if self.fusion_op_offset + length > len(self.coeffs_small) - 1:
                        sum_ovhds += self.coeffs_small[-1]
                    else:
                        sum_ovhds += self.coeffs_small[self.fusion_op_offset + length]
            else:
                sum_ovhds += self.coeffs_large[0]
                # sum_ovhds += self.coeffs_large[-1] * num_fused_ops
                for length in subop_lengths:
                    if self.fusion_op_offset + length > len(self.coeffs_large) - 1:
                        sum_ovhds += self.coeffs_large[-1]
                    else:
                        sum_ovhds += self.coeffs_large[self.fusion_op_offset + length]
        else:
            sum_ovhds += self.coeffs[0]
            # sum_ovhds += self.coeffs[-1] * num_fused_ops
            for length in subop_lengths:
                sum_ovhds += self.coeffs[self.fusion_op_offset + length]
        return sum_ovhds
    
    def dump(self, dump_path):
        with open(dump_path, "wb") as f:
            pickle.dump([self.max_dim, self.fusion_op_offset, 
                        self.regularization_lambda, self.use_dual_model, self.split_threshold,
                        self.coeffs, self.coeffs_large, self.coeffs_small, 
                        self.avg_elem_ovhd, self.avg_elem_ovhd_large, self.avg_elem_ovhd_small, 
                        self.A, self.A_normed, self.A_normed_large, self.A_normed_small, 
                        self.b, self.b_normed, self.b_normed_large, self.b_normed_small, 
                        self.elem_op_cache, self.dataset, self._dataset_path, 
                        self.module_details_dict, self.module_time_dict, 
                        self.row_dim, self.column_dim], f)
    
    def load(self, file_path):
        try:
            with open(file_path, "rb") as f:
                (self.max_dim, self.fusion_op_offset, self.regularization_lambda, self.use_dual_model, self.split_threshold,
                self.coeffs, self.coeffs_large, self.coeffs_small, 
                self.avg_elem_ovhd, self.avg_elem_ovhd_large, self.avg_elem_ovhd_small, 
                self.A, self.A_normed, self.A_normed_large, self.A_normed_small, 
                self.b, self.b_normed, self.b_normed_large, self.b_normed_small, 
                self.elem_op_cache, self.dataset, self._dataset_path, 
                self.module_details_dict, self.module_time_dict, 
                self.row_dim, self.column_dim) = pickle.load(f)
        except:
            self.use_dual_model = False
            self.split_threshold = -1
            self.regularization_lambda = 0.1
            with open(file_path, "rb") as f:
                (self.coeffs, self.avg_elem_ovhd, self.A, self.A_normed, self.b, self.b_normed, self.avg_elem_ovhd, self.elem_op_cache,
                self.dataset, self._dataset_path, self.module_details_dict, self.module_time_dict, self.row_dim, self.column_dim) = pickle.load(f)

class FusionKernelModel(Model):
    def __init__(self, op_code_vocab_size, op_code_embed_dim, subop_vocab_size, 
                    subop_embed_dim, node_embed_dim, lstm_output_dim, 
                    use_transform_hidden_layers, use_output_hidden_layers, dropout=0.2):
        super(FusionKernelModel, self).__init__()
        self.use_transform_hidden_layers = use_transform_hidden_layers
        self.use_output_hidden_layers = use_output_hidden_layers
        self.op_code_embedding = layers.Embedding(op_code_vocab_size, op_code_embed_dim, mask_zero=True)
        self.subop_embedding = layers.Embedding(subop_vocab_size, subop_embed_dim, mask_zero=True)
        self.mask_features = layers.Masking(mask_value=-1)
        self.embedding_transform = layers.Dense(node_embed_dim, activation="relu")
        self.embedding_hidden = layers.Dense(node_embed_dim, activation="relu")
        self.lstm = layers.Bidirectional(layers.LSTM(lstm_output_dim, return_sequences=True))
        self.output_hidden = layers.Dense(lstm_output_dim, activation="relu")
        self.output_transform = layers.Dense(1, activation=None)
        self.drop_out = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Shape of inputs:
        #   fusion_types: [batch_size, one_hot_dim (usually 2)]
        #   op_code_indexes: [batch_size, seq_length]
        #   op_hash_indexes: [batch_size, seq_length]
        #   features: [batch_size, seq_length, feature_length]
        fusion_types, op_code_indexes, op_hash_indexes, features = inputs

        # shape=[batch_size, seq_length, op_code_embed_dim]
        op_code_embedded = self.op_code_embedding(op_code_indexes) 

        # shape=[batch_size, seq_length, subop_embed_dim]
        op_hash_embedded = self.subop_embedding(op_hash_indexes)

        # shape same as features
        masked_features = self.mask_features(features)

        # shape=[batch_size, seq_length, op_code_embed_dim+subop_embed_dim+feature_length]
        input_feature_vec = layers.concatenate([op_code_embedded, op_hash_embedded, masked_features], axis=-1)

        # shape=[batch_size, seq_length, node_embed_dim]
        transformed_embedding = self.embedding_transform(input_feature_vec)
        transformed_embedding = self.drop_out(transformed_embedding, training=training)

        # shape = [batch_size, seq_length, node_embed_dim]
        if self.use_transform_hidden_layers:
            transformed_embedding = self.embedding_hidden(transformed_embedding)
            transformed_embedding = self.drop_out(transformed_embedding, training=training)

        # shape=[batch_size, seq_length, 2*lstm_output_dim]
        output_sequence = self.lstm(transformed_embedding)

        # shapes=[batch_size, 2*lstm_output_dim]
        sum_sequence = tf.reduce_sum(output_sequence, axis=1)
        max_sequence = tf.reduce_max(output_sequence, axis=1)
        mean_sequence = tf.reduce_mean(output_sequence, axis=1)

        # shape=[batch_size, 6*lstm_output_dim]
        final_embedding = layers.concatenate([fusion_types, sum_sequence, max_sequence, mean_sequence], axis=-1)

        # shape = [batch_size, lstm_output_dim]
        if self.use_output_hidden_layers:
            final_embedding = self.output_hidden(final_embedding)
            final_embedding = self.drop_out(final_embedding, training=training)

        # shape=[batch_size, 1]
        result = self.output_transform(final_embedding)
        return result

class BatchGeneratorBase():
    def __init__(self, dataset, batch_size):
        self.dataset_ = dataset
        self.batch_size = batch_size
        self.current_iter_ = 0
        self.dataset_size = dataset.train_size()
        self.re_init_()
    
    def shuffle_lists_(self, ls):
        z = list(zip(*ls))
        random.shuffle(z)
        return zip(*z)

    def re_init_(self):
        raise NotImplementedError
    
    def num_batches(self):
        return math.ceil(self.dataset_size / self.batch_size)

    def num_samples(self):
        return self.dataset_size
    
    def get_batch_selector_(self):
        if self.current_iter_ + self.batch_size >= len(self.op_codes_):
            effective_batch_size = len(self.op_codes_) - self.current_iter_
            epoch_end = True
        else:
            effective_batch_size = self.batch_size
            epoch_end = False
        def select_batch(x):
            return x[self.current_iter_:self.current_iter_ + effective_batch_size]
        return select_batch, epoch_end, effective_batch_size

    def next_batch(self):
        raise NotImplementedError

class BatchGenerator(BatchGeneratorBase):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size)

    def re_init_(self):
        self.fusion_types_, self.op_codes_, self.op_hashes_, self.feature_vectors_, self.labels_ = self.shuffle_lists_(self.dataset_.get_training_set())
        self.current_iter_ = 0

    def next_batch(self):
        select_batch, epoch_end, effective_batch_size = self.get_batch_selector_()
        fusion_types, op_codes, op_hashes, feature_vectors, labels = [select_batch(x) for x in [self.fusion_types_, self.op_codes_, self.op_hashes_, self.feature_vectors_, self.labels_]]
        # append padding
        max_sequence_length = max([len(v) for v in op_codes])
        op_codes = keras.preprocessing.sequence.pad_sequences(op_codes, padding="post")
        op_hashes = keras.preprocessing.sequence.pad_sequences(op_hashes, padding="post")
        # for feature vectors, we manually pad them with -1
        padded_feature_vectors = []
        for seq_vectors in feature_vectors:
            padded_seq_vectors = seq_vectors
            inner_dim = len(seq_vectors[0])
            if len(seq_vectors) < max_sequence_length:
                padded_seq_vectors += [[-1] * inner_dim] * (max_sequence_length - len(seq_vectors))
            padded_feature_vectors.append(padded_seq_vectors)
        padded_feature_vectors = np.array(padded_feature_vectors, dtype=np.float32)
        labels = np.array(labels)

        fusion_types = np.array(fusion_types, dtype=np.float32)

        self.current_iter_ += effective_batch_size

        if epoch_end:
            self.re_init_()

        return fusion_types, op_codes, op_hashes, padded_feature_vectors, labels, epoch_end

def train_kernel_model(dataset_path, save_dir, epochs=800, batch_size=64, 
                        op_code_embed_dim=16, subop_embed_dim=16, node_embed_dim=64, 
                        embedding_output_dim=32, use_embed_hidden=False, 
                        use_output_hidden=True, drop_out=0.2, learning_rate=5e-4,
                        early_stopping_patience=15, early_stopping_epsilon=1e-5):
    # load kernel dataset
    dataset_save_path = os.path.join(save_dir, "dataset.pickle")
    if os.path.exists(dataset_save_path):
        print("Loading kernel dataset from cache.")
        dataset = XlaKernelDataset(dataset_save_path)
    else:
        print("Loading kernel dataset...")
        dataset = XlaKernelDataset(dataset_path)
        dataset.dump_dataset(dataset_save_path)

    elem_op_cache_save_path = os.path.join(save_dir, "elem_op_cache.picke")
    ovhd_model_save_path = os.path.join(save_dir, "overhead.pickle")

    if os.path.exists(elem_op_cache_save_path):
        if not os.path.exists(ovhd_model_save_path):
            print("Loading elementary Op cache from cache.")
            elem_op_cache = ElementaryOpCache(load_from=elem_op_cache_save_path)
        else:
            elem_op_cache = None
            print("Overhead model and elementary op cache exists, doing nothing.")
    else:
        print("Loading elementary Op cache...")
        elem_op_cache = ElementaryOpCache(dataset_path)
        elem_op_cache.dump(elem_op_cache_save_path)
    
    if os.path.exists(ovhd_model_save_path):
        print("Overhead model exists, doing nothing.")
    else:
        # train module lvl model
        print("Loading module details...")
        module_overhead_model = XLAModuleOverheadModel(dataset_path, dataset, elem_op_cache)
        print("Fitting module overhead model...")
        module_overhead_model.fit()
        module_overhead_model.dump(ovhd_model_save_path)

    opcode_vocab_size = len(dataset.opcode2index) + 1
    ophash_vocab_size = len(dataset.ophash2index) + 1

    print("Training fused kernel model...")
    batch_generator = BatchGenerator(dataset, batch_size)
    model = FusionKernelModel(opcode_vocab_size, op_code_embed_dim, 
                    ophash_vocab_size, subop_embed_dim, 
                    node_embed_dim, embedding_output_dim,
                    use_embed_hidden, use_output_hidden, drop_out)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.mean_absolute_percentage_error
    metric = tf.keras.metrics.mean_absolute_percentage_error
    
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # write model config into file
    model_config = {"op_code_embed_dim": op_code_embed_dim, 
                    "subop_embed_dim": subop_embed_dim,
                    "node_embed_dim": node_embed_dim,
                    "embedding_output_dim": embedding_output_dim,
                    "use_embed_hidden": use_embed_hidden,
                    "use_output_hidden": use_output_hidden,
                    "drop_out": drop_out
                    }
    with open(os.path.join(save_dir, "model_config.pickle"), "wb") as f:
        pickle.dump(model_config, f)
    # training
    loss_history = []
    mape_history = []
    smoothed_val_history = []
    mape_val_history = []
    pbar = tqdm(total=batch_generator.num_batches() * epochs)
    epoch_loss_history = []
    epoch_mape_history = []
    early_stopping_counter = 0
    for epoch_num in range(epochs):
        while True:
            fusion_types, op_codes, op_hashes, padded_feature_vectors, labels, epoch_end = batch_generator.next_batch()
            loss, metric = model.train_on_batch(x=[fusion_types, op_codes, op_hashes, padded_feature_vectors], y=labels)
            epoch_loss_history.append(loss)
            epoch_mape_history.append(metric)
            pbar.update(1)
            if epoch_end:
                break
        if epoch_num % 10 == 0:
            # evaluate at the end of epoch
            predicted = []
            percentage_diff = []
            fusion_types_test, op_codes_test, op_hashes_test, feature_vectors_test, labels_test = dataset.get_test_set()
            for i in range(len(op_codes_test)):
                fusion_types = np.expand_dims(np.array(fusion_types_test[i], dtype=np.float32), 0)
                op_codes = np.expand_dims(np.array(op_codes_test[i]), 0)
                op_hashes = np.expand_dims(np.array(op_hashes_test[i]), 0)
                feature_vectors = np.expand_dims(np.array(feature_vectors_test[i], dtype=np.float32), 0)
                predicted_time = model.predict(x=[fusion_types, op_codes, op_hashes, feature_vectors])
                predicted.append(predicted_time)
                percentage_diff.append(np.abs(predicted_time - labels_test[i]) / labels_test[i])
            val_mape = np.average(percentage_diff) * 100
            mape_val_history.append(val_mape)
            if len(mape_val_history) > 1 and early_stopping_patience != 0:
                if mape_val_history[-2] - val_mape >= early_stopping_epsilon:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print("========== EARLY STOPPED ==========")
                        pbar.close()
                        break
                else:
                    early_stopping_counter = 0
        if epoch_num % 20 == 0:
            avg_mse = np.average(epoch_loss_history)
            avg_mape = np.average(epoch_mape_history)
            avg_val_mape = np.average(mape_val_history[-2:])
            tqdm.write("Epoch {}: MSE: {}, MAPE: {}, VAL MAPE:{}".format(epoch_num, avg_mse, avg_mape, avg_val_mape))
            loss_history.append(avg_mse)
            mape_history.append(avg_mape)
            smoothed_val_history.append(avg_val_mape)
            epoch_loss_history = []
            epoch_mape_history = []
    
            model_save_path = os.path.join(save_dir, "model_weights.h5")
            model.save_weights(model_save_path)
    
    model_save_path = os.path.join(save_dir, "model_weights.h5")
    model.save_weights(model_save_path)

    # evaluation
    predicted = []
    percentage_diff = []
    fusion_types_test, op_codes_test, op_hashes_test, feature_vectors_test, labels_test = dataset.get_test_set()
    for i in range(len(op_codes_test)):
        fusion_types = np.expand_dims(np.array(fusion_types_test[i], dtype=np.float32), 0)
        op_codes = np.expand_dims(np.array(op_codes_test[i]), 0)
        op_hashes = np.expand_dims(np.array(op_hashes_test[i]), 0)
        feature_vectors = np.expand_dims(np.array(feature_vectors_test[i], dtype=np.float32), 0)
        predicted_time = model.predict(x=[fusion_types, op_codes, op_hashes, feature_vectors])
        predicted.append(predicted_time)
        percentage_diff.append(np.abs(predicted_time - labels_test[i]) / labels_test[i])

    for i in range(10):
        print("predicted: {}, true: {}".format(predicted[i], labels_test[i]))

    print("Trained on {} samples, tested on {} samples.".format(batch_generator.num_samples(), len(predicted)))
    print("Test MAPE: {}, Median percentage difference: {}".format(np.average(percentage_diff), np.median(percentage_diff)))

# returns a model object which have a predict method
def load_kernel_model(model_save_dir):
    print("\n============== PROGRAM STARTS ==============\n")
    # load model config
    with open(os.path.join(model_save_dir, "model_config.pickle"), "rb") as f:
        model_config = pickle.load(f)
    
    dataset_save_path = os.path.join(model_save_dir, "dataset.pickle")
    training_dataset = XlaKernelDataset(dataset_save_path)

    print("\n============== Loading Elementary Op Cache ==============\n")
    elem_op_cache_save_path = os.path.join(model_save_dir, "elem_op_cache.picke")
    elem_op_cache = ElementaryOpCache(load_from=elem_op_cache_save_path)

    print("\n============== Loading Module Lvl Model ==============\n")
    overhead_model_save_path = os.path.join(model_save_dir, "overhead.pickle")
    ovhd_model = XLAModuleOverheadModel(load_from=overhead_model_save_path)

    print("\n============== Loading Keras Kernel Model ==============\n")
    opcode_vocab_size = len(training_dataset.opcode2index) + 1
    ophash_vocab_size = len(training_dataset.ophash2index) + 1
    model = FusionKernelModel(opcode_vocab_size, model_config["op_code_embed_dim"], 
                        ophash_vocab_size, model_config["subop_embed_dim"], 
                        model_config["node_embed_dim"], model_config["embedding_output_dim"],
                        model_config["use_embed_hidden"], model_config["use_output_hidden"], model_config["drop_out"])
    dummy_fusion_types = np.ones(shape=(1, training_dataset.max_fusion_type), dtype=np.float32) 
    dummy_op_codes = np.ones(shape=(1, 1), dtype=np.int32)
    dummy_op_hashes = np.ones(shape=(1, 1), dtype=np.int32)
    dummy_feature_vectors = np.ones(shape=(1, 1, training_dataset.feature_dim), dtype=np.float32)
    model([dummy_fusion_types, dummy_op_codes, dummy_op_hashes, dummy_feature_vectors])
    model_save_path = os.path.join(model_save_dir, "model_weights.h5")
    model.load_weights(model_save_path)
    return elem_op_cache, ovhd_model, model

class XLAModuleCostModel():
    def __init__(self, save_dir, tmp_dir = "./cost_model_tmp"):
        super().__init__()
        dataset_path = os.path.join(save_dir, "dataset.pickle")
        self.training_dataset = XlaKernelDataset(dataset_path)
        self.elem_op_cache, self.ovhd_model, self.kernel_model = load_kernel_model(save_dir)
        graph_def_path = os.path.join(save_dir, "graph_def.pickle")
        with open(graph_def_path, "rb") as f:
            self.graph_def = pickle.load(f)
        shape_dict_path = os.path.join(save_dir, "tensor_shapes.json")
        if not os.path.exists(shape_dict_path):
            shape_dict_path= None
        self.graph_def_util = GraphDefUtil(self.graph_def, shape_dict_path=shape_dict_path)
        self.computation_cache = {}
        # gutil = self.graph_def_util
        # code.interact(local=locals())
        self._tmp_dir = os.path.abspath(tmp_dir)
        if not os.path.isdir(self._tmp_dir):
            os.makedirs(self._tmp_dir)

    @classmethod
    def train_on_dataset(cls, dataset_path, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        shutil.copy(os.path.join(dataset_path, "tensor_shapes.json"), save_dir)
        dataset_folder_path = os.path.join(dataset_path, "dataset")
        train_kernel_model(dataset_folder_path, save_dir)
        graph_def_path = os.path.join(dataset_path, "cleaned_graph.json")
        with open(graph_def_path, "r") as f:
            cleaned_graph_def_str = f.read()
        graph_def = Parse(cleaned_graph_def_str, GraphDef())
        with open(os.path.join(save_dir, "graph_def.pickle"), "wb") as f:
            pickle.dump(graph_def, f)

    def predict(self, node_names):
        try:
            graph_def_path, config_path = self.graph_def_util.get_subgraph_def_config_from_nodes(node_names, self._tmp_dir, 0)
        except (GSNotInGraphError, GSSubgraphTooSmallError, GSNonFixedShapeError) as e:
            print("[Cost Model] Failed to generate legal graph def for input nodes: {}".format(e))
            return -1, {}
        except RuntimeError:
            traceback.print_exc()
            print("[Cost Model] Failed to generate legal graph def for input nodes.")
            return -1, {}

        if graph_def_path is None or config_path is None:
            if len(node_names) > 10:
                print("[Cost Model] Failed to predict the running time of ops: {} ...".format(list(node_names)[:10]))
            else:
                print("[Cost Model] Failed to predict the running time of ops: {} ...".format(node_names))
            return -1, {}
        unopt_hlo_path = os.path.join(self._tmp_dir, "unopt.txt")
        opt_hlo_path = os.path.join(self._tmp_dir, "opt.txt")
        feature_vec_path = os.path.join(self._tmp_dir, "feature.txt")
        try:
            compile_to_hlo(graph_def_path, config_path, unopt_hlo_path, opt_hlo_path)
            extract_kernel_features_from_hlo(opt_hlo_path, self._tmp_dir)

            config_path = os.path.join(self._tmp_dir, "module_config.txt")
            elem_op_hashes = []
            fused_op_infos = []
            with open(config_path, "r") as f:
                for line in f:
                    is_fused_op = bool(int(line.split(",")[0]))
                    if is_fused_op:
                        fused_op_hash = int(line.split(",")[1])
                        kernel_path = os.path.join(self._tmp_dir, line.split(",")[-1].split("/")[-1].strip())
                        kernel_sid = int(line.split(",")[-1].split("/")[-1].split(".txt")[0])
                        with open(kernel_path, "r") as f_kernel:
                            (adj, fusion_type, computation_hash, subop_infos) = self.training_dataset._parse_feature_file(f_kernel)
                        if (computation_hash != fused_op_hash):
                            print("Inconsistent hashes for kernel SID: {}".format(kernel_sid))
                            assert False
                        # generate representations
                        fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample = self.training_dataset.gen_representation_for_sample(fusion_type, subop_infos)
                        fused_op_infos.append((computation_hash, fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample))
                    else:
                        _, elem_op_hash, op_code = [v.strip() for v in line.split(",")]
                        elem_op_hashes.append((int(elem_op_hash), op_code))

            predicted_module_time = 0
            fused_kernel_sizes = [len(op_codes_in_sample) for (_, _, op_codes_in_sample, _, _) in fused_op_infos]

            breakdown_dict = {}
            breakdown_dict["elementary"] = {}
            breakdown_dict["fused"] = {}
            # overhead = ovhd_model.get_overhead(elem_op_hashes, len(fused_op_infos))
            overhead = self.ovhd_model.get_overhead(elem_op_hashes, fused_kernel_sizes)
            predicted_module_time += overhead
            breakdown_dict["overhead"] = overhead
            # print("[Cost Model] Predicted Overhead: {}".format(overhead))
            # look up elementary op cache
            for (hash_v, op_code) in elem_op_hashes:
                predicted_time, cache_hit = self.elem_op_cache.query(hash_v, op_code=op_code)
                predicted_module_time += predicted_time
                breakdown_dict["elementary"][hash_v] = predicted_time
                # print("[Cost Model] Predicted time for elem op hash {} (cache_hit: {} ,{}): {}".format(hash_v, cache_hit, op_code, predicted_time))
            
            # run model to predict fused kernel time
            for (computation_hash, fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample) in fused_op_infos:
                if computation_hash in self.computation_cache:
                    predicted_time = self.computation_cache[computation_hash]
                else:
                    fusion_types = np.expand_dims(np.array(fusion_type_one_hot, dtype=np.float32), 0)
                    op_codes = np.expand_dims(np.array(op_codes_in_sample), 0)
                    op_hashes = np.expand_dims(op_hashes_in_sample, 0)
                    feature_vectors = np.expand_dims(np.array(feature_vectors_in_sample, dtype=np.float32), 0)
                    predicted_time = self.kernel_model.predict(x=[fusion_types, op_codes, op_hashes, feature_vectors]).flatten()[0]
                    self.computation_cache[computation_hash] = predicted_time
                predicted_module_time += predicted_time
                breakdown_dict["fused"][computation_hash] = predicted_time
                # print("[Cost Model] Predicted time for fused kernel {}: {}".format(computation_hash, predicted_time))

        except Exception as e:
            traceback.print_exc()
            if len(node_names) > 10:
                print("[Cost Model] Failed to predict the running time of ops: {} ...".format(list(node_names)[:10]))
            else:
                print("[Cost Model] Failed to predict the running time of ops: {} ...".format(node_names))
            shutil.rmtree(self._tmp_dir)
            os.makedirs(self._tmp_dir)
            return -1, {}
        # clean up
        shutil.rmtree(self._tmp_dir)
        os.makedirs(self._tmp_dir)
        return predicted_module_time, breakdown_dict

    def execute(self, node_names):
        try:
            graph_def_path, config_path = self.graph_def_util.get_subgraph_def_config_from_nodes(node_names, self._tmp_dir, 0)
        except:
            traceback.print_exc()
            print("[Cost Model] Failed to generate legal graph def for input nodes. Possibly because an input node has non-fixed output shape.")
            return -1, {}
        if graph_def_path is None or config_path is None:
            if len(node_names) > 10:
                print("[Cost Model] Failed to predict the running time of ops: {} ...".format(list(node_names)[:10]))
            else:
                print("[Cost Model] Failed to predict the running time of ops: {} ...".format(node_names))
            return -1, {}
        unopt_hlo_path = os.path.join(self._tmp_dir, "unopt.txt")
        opt_hlo_path = os.path.join(self._tmp_dir, "opt.txt")
        feature_vec_path = os.path.join(self._tmp_dir, "feature.txt")
        try:
            compile_to_hlo(graph_def_path, config_path, unopt_hlo_path, opt_hlo_path)
            replay_and_generate_kernel_sample(0, unopt_hlo_path, self._tmp_dir, self._tmp_dir)
        except Exception as e:
            print(e)
            print("[Cost Model] Failed to execute the running time of ops: {}".format(node_names))
            return -1, {}
        exec_log_path = os.path.join(self._tmp_dir, "modules/0_exec.txt")
        exec_times = []
        with open(exec_log_path, "r") as f:
            for line in f:
                exec_times.append(float(line.strip()))
        module_exec_time = np.average(exec_times[10:-10])
        breakdown_path = os.path.join(self._tmp_dir, "modules/0_breakdown.txt")
        total_kernel_times = 0
        breakdown_dict = defaultdict(list)
        with open(breakdown_path, "r") as f:
            for line in f:
                hash_v = int(line.split(":")[0].strip())
                exec_time = float(line.split(":")[1].strip())
                breakdown_dict[hash_v].append(exec_time)
                total_kernel_times += exec_time
        overhead = module_exec_time - total_kernel_times
        breakdown_dict["overhead"] = overhead
        # clean up
        shutil.rmtree(self._tmp_dir)
        os.makedirs(self._tmp_dir)
        return module_exec_time, breakdown_dict
    
    def test_on_dataset(self, test_set_path):
        test_set_path = os.path.join(test_set_path, "dataset")
        training_dataset = self.training_dataset
        print("Loading test set...")
        test_dataset = XlaModuleTestSet(test_set_path, training_dataset)
        elem_op_cache = self.elem_op_cache
        ovhd_model = self.ovhd_model
        model = self.kernel_model

        print("\n==============Evaluation Starts==============\n")
        # evaluation
        predicted = []
        percentage_diff = []
        percentage_diff_large = []
        percentage_diff_small = []
        module_infos_as_list, labels_as_list, sample_ids = test_dataset.get_module_test_sets()
        
        undershoot_count = 0
        overshoot_count = 0
        small_count = 0
        large_count = 0
        small_time_count = 0
        large_time_count = 0
        large_error_sids = []
        for i in tqdm(random.sample(list(range(len(module_infos_as_list))), k=min(400, len(module_infos_as_list)))):
            predicted_module_time = 0
            elem_op_hashes, fused_op_infos = module_infos_as_list[i]

            fused_kernel_sizes = [len(op_codes_in_sample) for (_, op_codes_in_sample, _, _) in fused_op_infos]

            overhead = ovhd_model.get_overhead(elem_op_hashes, fused_kernel_sizes)
            predicted_module_time += overhead

            # look up elementary op cache
            for (hash_v, op_code) in elem_op_hashes:
                predicted_time, _ = elem_op_cache.query(hash_v, op_code=op_code)
                predicted_module_time += predicted_time
            
            # run model to predict fused kernel time
            for (fusion_type_one_hot, op_codes_in_sample, op_hashes_in_sample, feature_vectors_in_sample) in fused_op_infos:
                fusion_types = np.expand_dims(np.array(fusion_type_one_hot, dtype=np.float32), 0)
                op_codes = np.expand_dims(np.array(op_codes_in_sample), 0)
                op_hashes = np.expand_dims(op_hashes_in_sample, 0)
                feature_vectors = np.expand_dims(np.array(feature_vectors_in_sample, dtype=np.float32), 0)
                predicted_time = model.predict(x=[fusion_types, op_codes, op_hashes, feature_vectors]).flatten()[0]
                predicted_module_time += predicted_time
            predicted.append(predicted_module_time)
            if labels_as_list[i] > 500:
                percentage_diff_large.append(np.abs(predicted_module_time - labels_as_list[i]) / labels_as_list[i])
                large_count += 1
                large_time_count += labels_as_list[i]
            else:
                percentage_diff_small.append(np.abs(predicted_module_time - labels_as_list[i]) / labels_as_list[i])
                small_count += 1
                small_time_count += labels_as_list[i]
            percentage_diff.append(np.abs(predicted_module_time - labels_as_list[i]) / labels_as_list[i])
            if predicted_module_time - labels_as_list[i] >= 0:
                overshoot_count += 1
            else:
                undershoot_count += 1
            
            if np.abs(predicted_module_time - labels_as_list[i]) / labels_as_list[i] > 0.2:
                large_error_sids.append((sample_ids[i], predicted_module_time, labels_as_list[i]))
            
        with open("./large_error_sids.txt", "w") as f:
            for (sid, predicted_module_time, true_module_time) in large_error_sids:
                f.write(str(sid) + ", " + str(predicted_module_time) + ", " + str(true_module_time))
                f.write("\n")

        print("Tested on {} samples.".format(len(predicted)))
        print("Test MAPE: {}, Median percentage difference: {}".format(np.average(percentage_diff), np.median(percentage_diff)))
        print("Test MAPE on large: {}, Median percentage difference: {}".format(np.average(percentage_diff_large), np.median(percentage_diff_large)))
        print("Test MAPE on small: {}, Median percentage difference: {}".format(np.average(percentage_diff_small), np.median(percentage_diff_small)))
        print("Overshoot count: {}, undershoot count: {}".format(overshoot_count, undershoot_count))
        print("Small count: {}, large count: {}".format(small_count, large_count))
        print("Small time count: {}, large time count: {}".format(small_time_count, large_time_count))