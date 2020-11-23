from cost_model_xla.gen_samples import GDUNotInGraphError
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Model
import argparse
import shutil
import copy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.sparse as sp
import os
import pickle
import code
try:
    GraphDef = tf.GraphDef
except:
    GraphDef = tf.compat.v1.GraphDef
from cost_model_xla.xlatools import compile_to_hlo, extract_kernel_features_from_hlo, replay_and_generate_kernel_sample
from google.protobuf.json_format import Parse
from cost_model_xla.cost_model import XlaKernelDataset, XlaKernelGCNDataset, XLAModuleOverheadModel, ElementaryOpCache, XlaModuleTestSet, GraphDefUtil
from cost_model_xla.gen_samples import GDUNotInGraphError, GDUNonFixedShapeError, GDUSubgraphTooSmallError
from tqdm import tqdm, trange
from collections import defaultdict
import traceback

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

def normalize_adj(adj):
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    a_norm = adj.dot(d).transpose().dot(d).tocsr()
    return a_norm

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    return normalize_adj(adj + sp.eye(adj.shape[0]))

# from https://github.com/tkipf/keras-gcn/
class GraphConvolution(layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_kernel_gcn_model(fusion_type_dim, features_dim, op_code_vocab_size, op_code_embed_dim, subop_vocab_size, 
                subop_embed_dim, gcn_output_units, output_hidden_dim, dropout=0.2):
    # define layers
    op_code_embedding = layers.Embedding(op_code_vocab_size, op_code_embed_dim, name="embed_op_code")
    subop_embedding = layers.Embedding(subop_vocab_size, subop_embed_dim, name="embed_op_hash")
    GCN = GraphConvolution(gcn_output_units, activation="relu", name="GCN_1")
    GCN2 = GraphConvolution(gcn_output_units, activation="relu", name="GCN_2")
    output_hidden = layers.Dense(output_hidden_dim, activation="relu", name="output_hidden")
    output_transform = layers.Dense(1, activation=None, name="output_transform")


    # define inputs
    # inputs contains:
    #   fusion_types: (concated_graph_size, one_hot_dim (usually 2))
    #   op_code_indexes: (concated_graph_size)
    #   op_hash_indexes: (concated_graph_size)
    #   adj: (concated_graph_size, concated_graph_size), sparse block diagonal matrix
    #   features: (concated_graph_size, feature_length)
    #   graph_size_segments: (batch_size)
    fusion_types = layers.Input(shape=(fusion_type_dim,), name="fusion_types")
    op_code_indexes = layers.Input(shape=(), name="op_code")
    op_hash_indexes = layers.Input(shape=(), name="op_hash")
    adj = layers.Input(shape=(None,), sparse=True, name="adj_matrix")
    features = layers.Input(shape=(features_dim,), name="node_features")
    graph_size_segments = layers.Input(shape=(), dtype="int32", name="graph_sizes")

    # reshape indexes for embedding layer
    # shape=(1, concated_graph_size)
    op_code_reshaped = tf.expand_dims(op_code_indexes, axis=0, name="reshape_op_code")
    op_hash_reshaped = tf.expand_dims(op_hash_indexes, axis=0, name="reshape_op_hash")

    # shape=(1, concated_graph_size, op_code_embed_dim)
    op_code_embedded = op_code_embedding(op_code_reshaped) 
    # shape=(1, concated_graph_size, subop_embed_dim)
    op_hash_embedded = subop_embedding(op_hash_reshaped)

    # reshape them back
    # shape=(concated_graph_size, op_code_embed_dim)
    op_code_embedded = tf.squeeze(op_code_embedded, axis=[0], name="reshape_embed_op_code")
    # shape=(concated_graph_size, subop_embed_dim)
    op_hash_embedded = tf.squeeze(op_hash_embedded, axis=[0], name="reshape_embed_op_hash")

    # shape=(concated_graph_size, op_code_embed_dim+subop_embed_dim+feature_length)
    input_feature_vec = layers.concatenate([op_code_embedded, op_hash_embedded, features], axis=-1, name="input_vec_concat")

    # pass through gcn layers
    # shape=(concated_graph_size, gcn_output_units)
    gcn_output = GCN([input_feature_vec, adj])
    gcn_output = layers.Dropout(dropout, name="dropout_GCN_1")(gcn_output)
    gcn_output2 = GCN2([gcn_output, adj])
    gcn_output2 = layers.Dropout(dropout, name="dropout_GCN_2")(gcn_output2)

    # max pooling
    # shape=(batch_size, gcn_output_units)
    max_pooled_outputs = tf.math.segment_max(gcn_output2, graph_size_segments, name="max_pool")

    # avg pooling
    # shape=(batch_size, gcn_output_units)
    avg_pooled_outputs = tf.math.segment_mean(gcn_output2, graph_size_segments, name="avg_pool")

    # sum pooling
    # shape=(batch_size, gcn_output_units)
    sum_pooled_outputs = tf.math.segment_sum(gcn_output2, graph_size_segments, name="sum_pool")

    # graph_embedding
    # shape=[batch_size, fusion_type_len + 3*gcn_output_units]
    graph_embedding = layers.concatenate([fusion_types, max_pooled_outputs, avg_pooled_outputs, sum_pooled_outputs], axis=-1, name="output_concat")

    output = output_hidden(graph_embedding)
    output = layers.Dropout(dropout)(output)
    output = output_transform(output)

    return [fusion_types, op_code_indexes, op_hash_indexes, adj, features, graph_size_segments], output

# class FusionKernelGCN(Model):
#     def __init__(self, op_code_vocab_size, op_code_embed_dim, subop_vocab_size, 
#                 subop_embed_dim, gcn_output_units, output_hidden_dim, dropout=0.2):
#         super(FusionKernelGCN, self).__init__()
#         self.op_code_embedding = layers.Embedding(op_code_vocab_size, op_code_embed_dim)
#         self.subop_embedding = layers.Embedding(subop_vocab_size, subop_embed_dim)
#         self.GCN = GraphConvolution(gcn_output_units, activation="relu")
#         self.GCN2 = GraphConvolution(gcn_output_units, activation="relu")
#         self.output_hidden = layers.Dense(output_hidden_dim, activation="relu")
#         self.output_transform = layers.Dense(1, activation=None)
#         self.drop_out = layers.Dropout(dropout)
    
#     def call(self, inputs, training=False):
#         # inputs contains:
#         #   fusion_types: (concated_graph_size, one_hot_dim (usually 2))
#         #   op_code_indexes: (concated_graph_size)
#         #   op_hash_indexes: (concated_graph_size)
#         #   adj: (concated_graph_size, concated_graph_size), sparse block diagonal matrix
#         #   features: (concated_graph_size, feature_length)
#         #   graph_size_segments: (batch_size)
#         fusion_types, op_code_indexes, op_hash_indexes, adj, features, graph_size_segments = inputs

#         # reshape indexes for embedding layer
#         # shape=(1, concated_graph_size)
#         op_code_reshaped = tf.expand_dims(op_code_indexes, axis=0)
#         op_hash_reshaped = tf.expand_dims(op_hash_indexes, axis=0)

#         # shape=(1, concated_graph_size, op_code_embed_dim)
#         op_code_embedded = self.op_code_embedding(op_code_reshaped) 
#         # shape=(1, concated_graph_size, subop_embed_dim)
#         op_hash_embedded = self.subop_embedding(op_hash_reshaped)

#         # reshape them back
#         # shape=(concated_graph_size, op_code_embed_dim)
#         op_code_embedded = tf.squeeze(op_code_embedded)
#         # shape=(concated_graph_size, subop_embed_dim)
#         op_hash_embedded = tf.squeeze(op_hash_embedded)

#         # shape=(concated_graph_size, op_code_embed_dim+subop_embed_dim+feature_length)
#         input_feature_vec = layers.concatenate([op_code_embedded, op_hash_embedded, features], axis=-1)

#         # pass through gcn layers
#         # shape=(concated_graph_size, gcn_output_units)
#         gcn_output = self.GCN(input_feature_vec, adj)
#         gcn_output = self.drop_out(gcn_output)
#         gcn_output2 = self.GCN2(gcn_output, adj)
#         gcn_output2 = self.drop_out(gcn_output2)

#         # max pooling
#         # shape=(batch_size, gcn_output_units)
#         max_pooled_outputs = tf.math.segment_max(gcn_output2, graph_size_segments)

#         # avg pooling
#         # shape=(batch_size, gcn_output_units)
#         avg_pooled_outputs = tf.math.segment_mean(gcn_output2, graph_size_segments)

#         # sum pooling
#         # shape=(batch_size, gcn_output_units)
#         sum_pooled_outputs = tf.math.segment_sum(gcn_output2, graph_size_segments)

#         # graph_embedding
#         # shape=[batch_size, fusion_type_len + 3*gcn_output_units]
#         graph_embedding = layers.concatenate([fusion_types, max_pooled_outputs, avg_pooled_outputs, sum_pooled_outputs], axis=-1)

#         output = self.output_hidden(graph_embedding)
#         output = self.drop_out(output)
#         output = self.output_transform(output)

#         return output

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

class GCNBatchGenerator(BatchGeneratorBase):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size)
    
    def get_fusion_type_shape(self):
        return self.dataset_.max_fusion_type
    
    def get_feature_shape(self):
        return self.dataset_.feature_dim
    
    def re_init_(self):
        self.adjs_, self.fusion_types_, self.op_codes_, self.op_hashes_, self.feature_vectors_, self.labels_ = self.shuffle_lists_(self.dataset_.get_training_set())
        self.current_iter_ = 0
    
    def next_batch(self):
        select_batch, epoch_end, effective_batch_size = self.get_batch_selector_()
        adjs, fusion_types, op_codes, op_hashes, feature_vectors, labels = [select_batch(x) for x in [self.adjs_, self.fusion_types_, self.op_codes_, self.op_hashes_, self.feature_vectors_, self.labels_]]

        def flatten2D(arr):
            shapes = []
            flattened = []
            for elements in arr:
                shapes.append(len(elements))
                for ele in elements:
                    flattened.append(ele)
            return flattened, shapes
        # construct block diagonal adj matrix and normalize it
        block_matrix = sp.block_diag(adjs)
        normed_block_matrix = normalize_adj(block_matrix)

        labels = np.array(labels)
        fusion_types = np.array(fusion_types, dtype=np.float32)

        # needs flattening
        flat_op_codes, opc_shapes = flatten2D(op_codes)
        flat_op_codes = np.array(flat_op_codes)

        flat_op_hashes, oph_shapes = flatten2D(op_hashes)
        flat_op_hashes = np.array(flat_op_hashes)

        flat_op_vectors, opv_shapes = flatten2D(feature_vectors)
        flat_op_vectors = np.array(flat_op_vectors, dtype=np.float32)

        if not (opc_shapes == oph_shapes == opv_shapes):
            print("Expected equal shapes among op codes, op hashes and op feature vecs, but got {}".format([opc_shapes, oph_shapes, opv_shapes]))

        expanded_shapes = []
        for index, size in enumerate(opc_shapes):
            expanded_shapes += [index] * size
        graph_sizes = np.array(expanded_shapes, dtype=np.int32)

        self.current_iter_ += effective_batch_size

        if epoch_end:
            self.re_init_()

        return normed_block_matrix, fusion_types, flat_op_codes, flat_op_hashes, flat_op_vectors, labels, graph_sizes, epoch_end

def train_kernel_model(dataset_path, save_dir, epochs=800, batch_size=64, 
                        op_code_embed_dim=16, subop_embed_dim=16, node_embed_dim=64, 
                        embedding_output_dim=32, use_embed_hidden=False, 
                        use_output_hidden=True, drop_out=0.2, learning_rate=5e-4,
                        early_stopping_patience=15, early_stopping_epsilon=1e-5):
    # load kernel dataset
    dataset = XlaKernelDataset(dataset_path)
    # train module lvl model
    elem_op_cache = ElementaryOpCache(dataset_path)
    module_overhead_model = XLAModuleOverheadModel(dataset_path, dataset, elem_op_cache)
    module_overhead_model.fit()

    ovhd_model_save_path = os.path.join(save_dir, "overhead.pickle")
    module_overhead_model.dump(ovhd_model_save_path)
    
    dataset_save_path = os.path.join(save_dir, "dataset.pickle")
    dataset.dump_dataset(dataset_save_path)

    elem_op_cache_save_path = os.path.join(save_dir, "elem_op_cache.picke")
    elem_op_cache.dump(elem_op_cache_save_path)

    opcode_vocab_size = len(dataset.opcode2index) + 1
    ophash_vocab_size = len(dataset.ophash2index) + 1

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
    def __init__(self, save_dir, tmp_dir = "./cost_model_tmp", shape_dict_path=None):
        super().__init__()
        dataset_path = os.path.join(save_dir, "dataset.pickle")
        self.training_dataset = XlaKernelDataset(dataset_path)
        self.elem_op_cache, self.ovhd_model, self.kernel_model = load_kernel_model(save_dir)
        graph_def_path = os.path.join(save_dir, "graph_def.pickle")
        with open(graph_def_path, "rb") as f:
            self.graph_def = pickle.load(f)
        self.graph_def_util = GraphDefUtil(self.graph_def, shape_dict_path=shape_dict_path)
        self.computation_cache = {}
        # gutil = self.graph_def_util
        # code.interact(local=locals())
        self._tmp_dir = os.path.abspath(tmp_dir)
        if not os.path.isdir(self._tmp_dir):
            os.makedirs(self._tmp_dir)

    @classmethod
    def train_on_dataset(cls, dataset_path, graph_def, save_dir):
        train_kernel_model(dataset_path, save_dir)
        if isinstance(graph_def, str):
            with open(graph_def, "r") as f:
                cleaned_graph_def_str = f.read()
            graph_def = Parse(cleaned_graph_def_str, GraphDef())
        with open(os.path.join(save_dir, "graph_def.pickle"), "wb") as f:
            pickle.dump(graph_def, f)

    def predict(self, node_names):
        try:
            graph_def_path, config_path = self.graph_def_util.get_subgraph_def_config_from_nodes(node_names, self._tmp_dir, 0)
        except (GDUNotInGraphError, GDUSubgraphTooSmallError, GDUNonFixedShapeError) as e:
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
            os._exit(0)
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
        training_dataset = self.training_dataset
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
        for i in trange(min(len(module_infos_as_list), 400)):
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

        for i in range(10):
            print("predicted: {}, true: {}".format(predicted[i], labels_as_list[i]))

        print("Tested on {} samples.".format(len(predicted)))
        print("Test MAPE: {}, Median percentage difference: {}".format(np.average(percentage_diff), np.median(percentage_diff)))
        print("Test MAPE on large: {}, Median percentage difference: {}".format(np.average(percentage_diff_large), np.median(percentage_diff_large)))
        print("Test MAPE on small: {}, Median percentage difference: {}".format(np.average(percentage_diff_small), np.median(percentage_diff_small)))
        print("Overshoot count: {}, undershoot count: {}".format(overshoot_count, undershoot_count))
        print("Small count: {}, large count: {}".format(small_count, large_count))
        print("Small time count: {}, large time count: {}".format(small_time_count, large_time_count))