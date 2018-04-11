# -*- coding: utf-8 -*-


import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from provider import ptb_data_provider

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
provider = None
config = None
batch_size = 1024
size = 200
word_vocab_size = 216100
label_vocab_size = 50
initial_state = None
final_state = None

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEVICE_FLAG = "/gpu:0"

X = None
y = None

cell = None

cost = None
train_op = None
new_lr = None
lr_update = None


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.word_vocab_size = config['word_vocab_size']
        self.label_vocab_size = config['label_vocab_size']
        self._sequence_length = np.ones(self.batch_size, dtype=np.int16) * config["sequence_length"]
        self.is_training = is_training

    def calculate_loss(self, input_data):
        rnn_cell_list = []
        for i in range(config['num_layers']):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
            # rnn_cell = tf.contrib.rnn.GRUCell(size)

            if self.is_training and config['keep_prob'] < 1:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=config['keep_prob'])
            rnn_cell_list.append(rnn_cell)

        cell = tf.contrib.rnn.MultiRNNCell(rnn_cell_list, state_is_tuple=True)

        word_embedding = tf.get_variable("word_embedding", [self.word_vocab_size, self.hidden_size],
                                         dtype=data_type())
        label_embedding = tf.get_variable("label_embedding", [self.label_vocab_size, self.hidden_size],
                                          dtype=data_type())
        word_inputs = tf.nn.embedding_lookup(word_embedding, input_data[:, :2])
        label_inputs = tf.nn.embedding_lookup(label_embedding, input_data[:, 3:])

        if self.is_training and config['keep_prob'] < 1:
            word_inputs = tf.nn.dropout(word_inputs, config['keep_prob'])
            label_inputs = tf.nn.dropout(label_inputs, config['keep_prob'])

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
            [word_inputs[:, 0], label_inputs[:, 0], word_inputs[:, 1], label_inputs[:, 1]], axis=1)
        words_targets = tf.concat(
            [input_data[:, 1], input_data[:, 2]], axis=1)
        labels_targets = tf.concat([input_data[:, 3], input_data[:, 4]], axis=1)
        with tf.variable_scope("RNN"):
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=data_type(),
                sequence_length=self._sequence_length,
                inputs=inputs)
        # print(outputs)

        words_outputs = tf.concat([outputs[:, 1], outputs[:, 3]], axis=1)
        labels_outputs = tf.concat([outputs[:, 0], outputs[:, 2]], axis=1)

        words_outputs = tf.reshape(words_outputs, [-1, self.hidden_size])
        labels_outputs = tf.reshape(labels_outputs, [-1, self.hidden_size])

        word_softmax_w = tf.get_variable(
            "word_softmax_w", [self.hidden_size, self.word_vocab_size], dtype=data_type())
        word_softmax_b = tf.get_variable("word_softmax_b", [self.word_vocab_size], dtype=data_type())

        label_softmax_w = tf.get_variable(
            "label_softmax_w", [self.hidden_size, self.label_vocab_size], dtype=data_type())
        label_softmax_b = tf.get_variable("label_softmax_b", [self.label_vocab_size], dtype=data_type())

        words_y_flat = tf.reshape(words_outputs, [-1])
        labels_y_flat = tf.reshape(labels_outputs, [-1])
        words_logits = tf.matmul(words_outputs, word_softmax_w) + word_softmax_b
        labels_logits = tf.matmul(labels_outputs, label_softmax_w) + label_softmax_b
        words_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=words_logits, labels=words_y_flat)
        labels_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=labels_logits, labels=labels_y_flat)

        cost = tf.reduce_mean([words_losses, labels_losses])
        return cost

    def update_model(self, cost):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config['max_grad_norm'])
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


# @make_spin(Spin1, "Running epoch...")
def run_epoch(session, model, provider, status, verbose=False, restore_type=0, restored_model_dir="",
              eval_op=tf.no_op()):
    """Runs the model on the given data."""
    start_time = time.time()
    stage_time = time.time()
    costs = 0.0
    iters = 0
    words = 0
    provider.status = status
    first_flag = True
    for data_tensor, batch_words_num in provider():
        data_flag = True
        epoch_size = provider.get_current_epoch_size()
        sub_iters = 0
        while data_flag:
            try:
                with tf.device(DEVICE_FLAG):
                    cost_op = model.calculate_loss(data_tensor)
                if status == 'train':
                    model.update_model(cost_op)
                if first_flag:
                    session.run(tf.global_variables_initializer())
                    if restore_type == 1:
                        new_saver = tf.train.Saver()
                        new_saver.restore(session, tf.train.latest_checkpoint(
                            restored_model_dir))
                    for v in tf.global_variables():
                        print(v.name)
                    model.assign_lr(session, config['learning_rate'])
                    session.run(model.lr)
                    first_flag = False
                cost, _ = session.run([cost_op, eval_op])
                costs += cost
                words += batch_words_num
                iters += 1
                sub_iters += 1

                if iters % 1000 == 0:
                    print("current_loss: %.3f" % cost)
                divider = epoch_size // 100
                divider_10 = epoch_size // 10
                if divider == 0:
                    divider = 1
                if verbose and sub_iters % divider == 0:
                    if not sub_iters % divider_10 == 0:
                        print("         %.3f perplexity: %.3f time cost: %.3fs" %
                              (sub_iters * 1.0 / epoch_size, np.exp(costs / iters),
                               time.time() - stage_time), end='\r')
                if verbose and sub_iters % divider_10 == 0:
                    print("%.3f perplexity: %.3f speed: %.0f wps time cost: %.3fs" %
                          (sub_iters * 1.0 / epoch_size, np.exp(costs / iters),
                           words * model.batch_size / (time.time() - start_time), time.time() - stage_time))
                    stage_time = time.time()

            except tf.errors.OutOfRangeError:
                data_flag = False
    return np.exp(costs / iters)


def main():
    provider = ptb_data_provider()
    provider.status = 'train'
    config = provider.get_config()
    eval_config = config.copy()
    eval_config['batch_size'] = 1
    model_dir = config["model_dir"]
    train_type = config.get("train_type", 0)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # print (config)
    # print (eval_config)
    session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mdev = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()
        for i in range(config['max_max_epoch']):
            print("Epoch: %d" % i)
            print("Starting Time:", datetime.now())
            train_perplexity = run_epoch(session, m, provider, 'train', verbose=True, restore_type=train_type,
                                         restored_model_dir=model_dir, eval_op=m.train_op)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            print("Ending Time:", datetime.now())
            save_path = saver.save(session, './model/misscut_rnn_model', global_step=i)
            print("Model saved in file: %s" % save_path)
            print("Starting Time:", datetime.now())
            dev_perplexity = run_epoch(session, mdev, provider, 'dev')
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, dev_perplexity))
            print("Ending Time:", datetime.now())
            if (i % 13 == 0 and not i == 0):
                print("Starting Time:", datetime.now())
                test_perplexity = run_epoch(session, mtest, provider, 'test')
                print("Test Perplexity: %.3f" % test_perplexity)
                print("Ending Time:", datetime.now())

        test_perplexity = run_epoch(session, mtest, provider, 'test')
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()
