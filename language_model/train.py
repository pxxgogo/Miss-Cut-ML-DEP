# -*- coding: utf-8 -*-


import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from .provider import Data_provider
import json

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

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split.append(fn(**{k: v[i] for k, v in in_splits.items()}))

    return tf.reduce_mean(out_split)


class Model(object):
    """The PTB model."""

    def __init__(self, config, state):
        self._batch_size = config['batch_size']
        self._hidden_size = config['hidden_size']
        self._word_vocab_size = config['word_vocab_size']
        self._label_vocab_size = config['label_vocab_size']
        self._config = config
        self._state = state
        self._data_placeholder = tf.placeholder(tf.int32, [None, self._config["sequence_length"]])
        if state == 'train':
            self._dataset = tf.data.Dataset.from_tensor_slices(self._data_placeholder)
            self._dataset = self._dataset.shuffle(buffer_size=100000).apply(
                tf.contrib.data.batch_and_drop_remainder(self._batch_size))
            self._dataset_iterator = self._dataset.make_initializable_iterator()
            self._cost_op = make_parallel(self.calculate_cost, config["gpu_num"],
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
            self._cost_op = make_parallel(self.calculate_cost, config["gpu_num"],
                                          input_data=self._dataset_iterator.get_next())

        else:
            self._dataset = tf.data.Dataset.from_tensor_slices(self._data_placeholder)
            self._dataset = self._dataset.shuffle(buffer_size=100000)
            self._dataset_iterator = self._dataset.make_initializable_iterator()
            data_tensor = tf.reshape(self._dataset_iterator.get_next(), [1, -1])
            self._cost_op = self.calculate_cost(data_tensor)

    def calculate_cost(self, input_data):

        rnn_cell_list = []
        for nn_info in range(self._config['layer_num']):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(self._hidden_size, forget_bias=0.0, state_is_tuple=True)
            # rnn_cell = tf.contrib.rnn.GRUCell(size)

            if self._state == 'train' and self._config['keep_prob'] < 1:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self._config['keep_prob'])
            rnn_cell_list.append(rnn_cell)

        cell = tf.contrib.rnn.MultiRNNCell(rnn_cell_list, state_is_tuple=True)
        word_embedding = tf.get_variable("word_embedding", [self._word_vocab_size, self._hidden_size],
                                         dtype=data_type())
        label_embedding = tf.get_variable("label_embedding", [self._label_vocab_size, self._hidden_size],
                                          dtype=data_type())
        word_inputs = tf.nn.embedding_lookup(word_embedding, input_data[:, :2])
        label_inputs = tf.nn.embedding_lookup(label_embedding, input_data[:, 3:])

        if self._state == 'train' and self._config['keep_prob'] < 1:
            word_inputs = tf.nn.dropout(word_inputs, self._config['keep_prob'])
            label_inputs = tf.nn.dropout(label_inputs, self._config['keep_prob'])

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        inputs = tf.concat(
            [word_inputs[:, 0:1], label_inputs[:, 0:1], word_inputs[:, 1:2], label_inputs[:, 1:2]], axis=1)
        words_targets = input_data[:, 1:3]
        labels_targets = input_data[:, 3:5]
        sequence_length = np.ones(input_data.shape[0], dtype=np.int16) * self._config["sequence_length"]
        with tf.variable_scope("RNN"):
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=data_type(),
                sequence_length=sequence_length,
                inputs=inputs)
        # print(outputs)

        words_outputs = tf.concat([outputs[:, 1:2], outputs[:, 3:4]], axis=1)
        labels_outputs = tf.concat([outputs[:, 0:1], outputs[:, 2:3]], axis=1)
        # print(words_outputs.shape, labels_outputs.shape)

        words_outputs = tf.reshape(words_outputs, [-1, self._hidden_size])
        labels_outputs = tf.reshape(labels_outputs, [-1, self._hidden_size])

        word_softmax_w = tf.get_variable(
            "word_softmax_w", [self._hidden_size, self._word_vocab_size], dtype=data_type())
        word_softmax_b = tf.get_variable("word_softmax_b", [self._word_vocab_size], dtype=data_type())

        label_softmax_w = tf.get_variable(
            "label_softmax_w", [self._hidden_size, self._label_vocab_size], dtype=data_type())
        label_softmax_b = tf.get_variable("label_softmax_b", [self._label_vocab_size], dtype=data_type())

        words_y_flat = tf.reshape(words_targets, [-1])
        labels_y_flat = tf.reshape(labels_targets, [-1])
        words_logits = tf.matmul(words_outputs, word_softmax_w) + word_softmax_b
        labels_logits = tf.matmul(labels_outputs, label_softmax_w) + label_softmax_b
        # print(words_logits, words_targets)
        # print(labels_logits, labels_targets)
        words_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=words_logits, labels=words_y_flat)
        labels_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=labels_logits, labels=labels_y_flat)

        cost = tf.reduce_mean([words_losses, labels_losses])
        return cost

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
    def data_placeholder(self):
        return self._data_placeholder


# @make_spin(Spin1, "Running epoch...")
def run_epoch(session, models, provider, status, config, verbose=False):
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
    saver = tf.train.Saver()
    saver_No = 0
    eval_config = config.copy()
    eval_config['batch_size'] = 1
    if status == 'train':
        critical_model = models[0]
    elif status == 'dev':
        critical_model = models[1]
    else:
        critical_model = models[2]
    for data, batch_words_num in provider():
        data_flag = True
        epoch_size = provider.get_current_epoch_size()
        sub_iters = 0
        session.run(critical_model.dataset_iterator.initializer, feed_dict={critical_model.data_placeholder: data})
        while data_flag:
            # print(sub_iters)
            try:
                if status == "train":
                    eval_op = critical_model.train_op
                cost, _ = session.run([critical_model.cost_op, eval_op])
                # print(cost)
                costs += cost
                words += batch_words_num
                iters += 1
                sub_iters += 1
                if iters % 1000 == 0:
                    print("current_loss: %.3f" % cost, end='\r')
                divider = epoch_size // 100
                divider_10 = epoch_size // 10
                if divider == 0:
                    divider = 1
                if verbose and sub_iters % divider == 0:
                    if not sub_iters % divider_10 == 0:
                        print("                         %.3f perplexity: %.3f time cost: %.3fs" %
                              (sub_iters * 1.0 / epoch_size, np.exp(costs / iters),
                               time.time() - stage_time), end='\r')
                if verbose and sub_iters % divider_10 == 0:
                    print("%.3f perplexity: %.3f speed: %.0f wps time cost: %.3fs" %
                          (sub_iters * 1.0 / epoch_size, np.exp(costs / iters),
                           words * config["batch_size"] / (time.time() - start_time), time.time() - stage_time))
                    stage_time = time.time()
            except tf.errors.OutOfRangeError:
                data_flag = False
            if not data_flag and status == 'train':
                save_path = saver.save(session, os.path.join(config["model_dir"], 'misscut_model'),
                                       global_step=saver_No)
                saver_No += 1
                print("Model saved in file: %s" % save_path)
                dev_provider = Data_provider(config)
                print("Starting Time:", datetime.now())
                dev_perplexity = run_epoch(session, models, dev_provider, 'dev',
                                           config)
                print("Valid Perplexity: %.3f" % dev_perplexity)
                print("Ending Time:", datetime.now())
    if status == "train":
        print("Starting Time:", datetime.now())
        test_perplexity = run_epoch(session, models, provider, 'test', eval_config)
        print("Test Perplexity: %.3f" % test_perplexity)
        print("Ending Time:", datetime.now())
    return np.exp(costs / iters)


CONFIG_FILE_NAME = "./language_model/config.json"


def main():
    init_config = json.load(open(CONFIG_FILE_NAME))
    provider = Data_provider(init_config)
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
        for i in range(config['max_max_epoch']):
            m.assign_lr(session, config['learning_rate'])
            session.run(m.lr)
            print("Epoch: %d" % i)
            print("Starting Time:", datetime.now())
            train_perplexity = run_epoch(session, (m, mdev, mtest), provider, 'train', config, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            print("Ending Time:", datetime.now())
        test_perplexity = run_epoch(session, (m, mdev, mtest), provider, 'test', eval_config)
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()
