# -*- coding: utf-8 -*-


import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from .provider import Data_provider

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)
    costs = []
    error_nums = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                cost, error_num = fn(**{k: v[i] for k, v in in_splits.items()})
                costs.append(cost)
                error_nums.append(error_num)
    return tf.reduce_mean(costs), tf.reduce_sum(error_nums)


class Model(object):
    def __init__(self, config, state):
        self._batch_size = config['batch_size']
        self._hidden_size = config['hidden_size']
        self._word_vocab_size = config['word_vocab_size']
        self._label_vocab_size = config['label_vocab_size']
        self._config = config
        self._state = state
        self._data_placeholder = tf.placeholder(tf.int32,
                                                [None, self._config["sequence_length"] + self._config["output_size"]])
        if state == 'train':
            self._dataset = tf.data.Dataset.from_tensor_slices(self._data_placeholder)
            self._dataset = self._dataset.shuffle(buffer_size=100000).apply(
                tf.contrib.data.batch_and_drop_remainder(self._batch_size))
            self._dataset_iterator = self._dataset.make_initializable_iterator()
            self._cost_op, self._error_num_op = make_parallel(self.calculate_cost, config["gpu_num"],
                                                              input_data=self._dataset_iterator.get_next())
            # self._cost_op = self.calculate_cost(self._dataset_iterator.get_next())
            # with tf.device("/cpu:0"):
            self.update_model(self._cost_op)

        elif state == 'dev':
            self._dataset = tf.data.Dataset.from_tensor_slices(self._data_placeholder)
            self._dataset = self._dataset.shuffle(buffer_size=100000).apply(
                tf.contrib.data.batch_and_drop_remainder(self._batch_size))
            self._dataset_iterator = self._dataset.make_initializable_iterator()
            # self._cost_op = self.calculate_cost(self._dataset_iterator.get_next())
            self._cost_op, self._error_num_op = make_parallel(self.calculate_cost, config["gpu_num"],
                                                              input_data=self._dataset_iterator.get_next())

        else:
            self._dataset = tf.data.Dataset.from_tensor_slices(self._data_placeholder)
            self._dataset = self._dataset.shuffle(buffer_size=100000)
            self._dataset_iterator = self._dataset.make_initializable_iterator()
            data_tensor = tf.reshape(self._dataset_iterator.get_next(), [1, -1])
            self._cost_op, self._error_num_op = self.calculate_cost(data_tensor)

    def calculate_cost(self, input_data):
        sub_batch_size = self._batch_size // self._config["gpu_num"]
        word_embedding = tf.get_variable("word_embedding", [self._word_vocab_size, self._hidden_size],
                                         dtype=data_type())
        label_embedding = tf.get_variable("label_embedding", [self._label_vocab_size, self._hidden_size],
                                          dtype=data_type())
        word_inputs = tf.nn.embedding_lookup(word_embedding, input_data[:, :3])
        label_inputs = tf.nn.embedding_lookup(label_embedding, input_data[:, 3:5])
        labels = input_data[:, 5:]

        if self._state == 'train' and self._config['keep_prob'] < 1:
            word_inputs = tf.nn.dropout(word_inputs, self._config['keep_prob'])
            label_inputs = tf.nn.dropout(label_inputs, self._config['keep_prob'])

        data = tf.concat(
            [word_inputs[:, 0:1], label_inputs[:, 0:1], word_inputs[:, 1:2], label_inputs[:, 1:2], word_inputs[:, 2:3]],
            axis=1)
        nn_infos = self._config['nn_infos']
        layer_No = 0
        for nn_info in nn_infos:
            if nn_info["net_type"] == "CONV":
                if len(data.shape) == 3:
                    data = tf.reshape(data, [data.shape[0], data.shape[1], data.shape[2], 1])
                for i in range(nn_info["repeated_times"]):
                    data = self.add_conv_layer(layer_No, data, nn_info["filter_size"], nn_info["out_channels"],
                                               nn_info["filter_type"], self._config["regularized_lambda"],
                                               self._config["regularized_flag"])
                    layer_No += 1
            elif nn_info["net_type"] == "POOL":
                if len(data.shape) == 3:
                    data = tf.reshape(data, [data.shape[0], data.shape[1], data.shape[2], 1])
                for i in range(nn_info["repeated_times"]):
                    data = self.add_pool_layer(layer_No, data, nn_info["pool_size"], nn_info["pool_type"])
                    layer_No += 1
            elif nn_info["net_type"] == "DENSE":
                for i in range(nn_info["repeated_times"]):
                    data = self.add_dense_layer(layer_No, data, nn_info["output_size"], nn_info["keep_prob"],
                                                self._config["regularized_lambda"], self._config["regularized_flag"])
                    layer_No += 1

        data = tf.reshape(data, [sub_batch_size, -1])
        softmax_w = tf.get_variable(
            "softmax_w", [data.shape[1], 2], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [2], dtype=data_type())
        logits = tf.matmul(data, softmax_w) + softmax_b
        rets = tf.argmax(logits, axis=1)
        tags = tf.argmax(labels, axis=1)
        error_num = tf.reduce_sum(tf.abs(rets - tags))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        cost = tf.reduce_mean(loss)
        return cost, error_num

    def add_conv_layer(self, No, input, filter_size, out_channels, filter_type, regularized_lambda, r_flag=True,
                       strides=[1, 1, 1, 1]):
        with tf.variable_scope("conv_layer_%d" % No):
            W = tf.get_variable('filter', [filter_size[0], filter_size[1], input.shape[3], out_channels])
            if r_flag:
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularized_lambda)(W))
            b = tf.get_variable('bias', [out_channels])
            conv = tf.nn.conv2d(
                input,
                W,
                strides=strides,
                padding=filter_type,
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        return h

    def add_pool_layer(self, No, input, pool_size, pool_type, strides=[1, 1, 1, 1]):
        for i in range(2):
            if pool_size[i] == -1:
                pool_size[i] = input.shape[1 + i]
        with tf.variable_scope("pool_layer_%d" % No):
            pooled = tf.nn.max_pool(
                input,
                ksize=[1, pool_size[0], pool_size[1], 1],
                padding=pool_type,
                strides=strides,
                name='pool'
            )
        return pooled

    def get_length(self, input):
        ret = 1
        for i in range(1, len(input.shape)):
            ret *= int(input.shape[i])
        return ret

    def add_dense_layer(self, No, input, output_size, keep_prob, regularized_lambda, r_flag=True):
        with tf.variable_scope("dense_layer_%d" % No):
            input_length = self.get_length(input)
            W = tf.get_variable('dense', [input_length, output_size])
            if r_flag:
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularized_lambda)(W))
            b = tf.get_variable('bias', [output_size])
            data = tf.reshape(input, [-1, int(input_length)])
            data = tf.nn.relu(tf.matmul(data, W) + b)
            if keep_prob < 1.0:
                data = tf.nn.dropout(data, keep_prob)
        return data

    def update_model(self, cost):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars, colocate_gradients_with_ops=True),
                                          self._config['max_grad_norm'])
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def train_op(self):
        return self._train_op

    @property
    def lr(self):
        return self._lr

    @property
    def dataset_iterator(self):
        return self._dataset_iterator

    @property
    def cost_op(self):
        return self._cost_op

    @property
    def error_num_op(self):
        return self._error_num_op

    @property
    def data_placeholder(self):
        return self._data_placeholder


# @make_spin(Spin1, "Running epoch...")
def run_epoch(session, model, provider, status, config, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    stage_time = time.time()
    costs = 0.0
    iters = 0
    words = 0
    eval_op = tf.no_op()
    provider.status = status
    batch_size = config["batch_size"]
    sum = 0
    correct_sum = 0
    for data, batch_words_num in provider():
        data_flag = True
        epoch_size = provider.get_current_epoch_size()
        sub_iters = 0
        session.run(model.dataset_iterator.initializer, feed_dict={model.data_placeholder: data})
        while data_flag:
            # print(sub_iters)
            try:
                if status == "train":
                    eval_op = model.train_op
                cost, error_num, _ = session.run(
                    [model.cost_op, model.error_num_op, eval_op])
                # print(cost)
                costs += cost
                tmp_precision = (batch_size - error_num) / batch_size
                sum += batch_size
                correct_sum += (batch_size - error_num)
                words += batch_words_num
                iters += 1
                sub_iters += 1
                if iters % 1000 == 0:
                    print("%.3f, %.3f" % (cost, tmp_precision), end='\r')
                divider = epoch_size // 100
                divider_10 = epoch_size // 10
                if divider == 0:
                    divider = 1
                if verbose and sub_iters % divider == 0:
                    if not sub_iters % divider_10 == 0:
                        print("                  %.3f perplexity: %.3f time cost: %.3fs, precision: %.3f" %
                              (sub_iters * 1.0 / epoch_size, np.exp(costs / iters),
                               time.time() - stage_time, (correct_sum / sum)), end='\r')
                if verbose and sub_iters % divider_10 == 0:
                    print("%.3f perplexity: %.3f speed: %.0f wps time cost: %.3fs, precision: %.3f" %
                          (sub_iters * 1.0 / epoch_size, np.exp(costs / iters),
                           words * config["batch_size"] / (time.time() - start_time), time.time() - stage_time,
                           (correct_sum / sum)))
                    stage_time = time.time()
            except tf.errors.OutOfRangeError:
                data_flag = False
    return np.exp(costs / iters), correct_sum / sum


def main():
    provider = Data_provider()
    provider.status = 'train'
    config = provider.get_config()
    eval_config = config.copy()
    eval_config['batch_size'] = 1
    model_dir = config["model_dir"]
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    restored_type = config["restored_type"]

    # print (config)
    # print (eval_config)
    session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(config=config, state='train')
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mdev = Model(config=config, state='dev')
            mtest = Model(config=eval_config, state='test')

        session.run(tf.global_variables_initializer())
        if restored_type == 1:
            new_saver = tf.train.Saver()
            new_saver.restore(session, tf.train.latest_checkpoint(
                config["model_dir"]))
        for v in tf.global_variables():
            print(v.name)
        saver = tf.train.Saver()
        for i in range(config['max_max_epoch']):
            m.assign_lr(session, config['learning_rate'])
            session.run(m.lr)
            print("Epoch: %d" % i)
            print("Starting Time:", datetime.now())
            train_perplexity, precision = run_epoch(session, m, provider, 'train', config, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f, Precision: %.3f" % (i + 1, train_perplexity, precision))
            print("Ending Time:", datetime.now())
            save_path = saver.save(session, os.path.join(model_dir, 'misscut_model'), global_step=i)
            print("Model saved in file: %s" % save_path)
            print("Starting Time:", datetime.now())
            dev_perplexity, precision = run_epoch(session, mdev, provider, 'dev', config)
            print("Epoch: %d Valid Perplexity: %.3f, Precision: %.3f" % (i + 1, dev_perplexity, precision))
            print("Ending Time:", datetime.now())
            if (i % 13 == 0 and not i == 0):
                print("Starting Time:", datetime.now())
                test_perplexity, precision = run_epoch(session, mtest, provider, 'test', eval_config)
                print("Test Perplexity: %.3f, Precision: %.3f" % (test_perplexity, precision))
                print("Ending Time:", datetime.now())

        test_perplexity, precision = run_epoch(session, mtest, provider, 'test', eval_config)
        print("Test Perplexity: %.3f, Precision: %.3f" % (test_perplexity, precision))


if __name__ == "__main__":
    main()
