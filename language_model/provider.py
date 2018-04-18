import json
import os
import traceback
from os import path as op

import numpy as np
import random
import tensorflow as tf

# from memory_profiler import profile

CONFIG_FILE_NAME = "config.json"


class ptb_data_provider(object):

    def __init__(self):
        self.data_dir = ''

        self.input_compat = input
        self.data_dir = ''
        self.model = ''
        self.status = 'IDLE'

        self.batch_size = 1
        self.yield_pos = [0, 0, 0]
        self.training_corpus_paths = []
        self.training_corpus_num = 0
        self.test_corpus_path = ""
        self.dev_corpus_path = ""
        self.current_epoch_size = -1
        self._parse_config()

    def _parse_config(self):
        try:
            assert op.isfile(CONFIG_FILE_NAME) and os.access(CONFIG_FILE_NAME, os.R_OK)
            self.config = json.load(open(CONFIG_FILE_NAME, 'r'))
            assert 'data_dir' in self.config
            self.data_dir = self.config['data_dir']
            assert op.isdir(self.data_dir) and os.access(self.data_dir, os.R_OK)
            self.filenames = os.listdir(self.data_dir)
            self.batch_size = self.config["batch_size"]
            for filename in self.filenames:
                if not filename.endswith("dat.npy"):
                    continue
                fullname = op.join(self.data_dir, filename)
                print(fullname)
                assert op.isfile(fullname) and os.access(fullname, os.R_OK)
                if fullname.endswith(self.config["dev_corpus_suffix"]):
                    self.dev_corpus_path = fullname
                elif fullname.endswith(self.config["test_corpus_suffix"]):
                    self.test_corpus_path = fullname
                else:
                    self.training_corpus_paths.append(fullname)
            self.training_corpus_num = len(self.training_corpus_paths)
        except AssertionError:
            print(traceback.print_exc())
            self.data_dir = ''
            self.model = ''
            self.threshold = -1
            print("Configure file load failed.")
            cond = False
            exit()
        finally:
            print("OK")

    def _test_path(self):
        if len(self.data_dir) == 0:
            return False, "Data path length should be greater than 0."
        elif not op.isdir(self.data_dir):
            return False, "This path is not a directory."
        elif not os.access(self.data_dir, os.R_OK):
            return False, "This path is not accessible."
        else:
            for filename in self.filenames:
                fullname = op.join(self.data_dir, filename)
                if not op.isfile(fullname):
                    return False, "File {} not found.".format(filename)
                elif not os.access(fullname, os.R_OK):
                    return False, "File {} not accessible"
        return True, "Accepted"



    def get_epoch_size(self, data):
        if self.status == 'train':
            return (data.shape[0]) // self.batch_size - 1
        elif self.status == 'dev':
            return (data.shape[0]) // self.batch_size - 1
        elif self.status == 'test':
            return data.shape[0] - 1
        else:
            return None

    def get_config(self):
        return self.config

    def get_current_epoch_size(self):
        return self.current_epoch_size

    def init_training_corpus(self):
        random.shuffle(self.training_corpus_paths)

    def get_training_data(self):
        training_corpus_index = 0
        batch_words_num = self.batch_size * self.config["sequence_length"]
        while training_corpus_index < self.training_corpus_num:
            training_corpus_path = self.training_corpus_paths[training_corpus_index]
            training_data_np = np.load(training_corpus_path)
            self.current_epoch_size = self.get_epoch_size(training_data_np)
            print("TRAINING_FILE_PATH: %s, EPOCH_SIZE: %d" % (training_corpus_path, self.current_epoch_size))
            yield training_data_np, batch_words_num
            training_corpus_index += 1



    def get_dev_test_data(self):
        if self.status == "dev":
            batch_words_num = self.batch_size * self.config["sequence_length"]
            corpus_data_np = np.load(self.dev_corpus_path)
            self.current_epoch_size = self.get_epoch_size(corpus_data_np)
            print("DEV_FILE_PATH: %s, EPOCH_SIZE: %d" % (self.dev_corpus_path, self.current_epoch_size))
            yield corpus_data_np, batch_words_num

        else:
            batch_words_num = self.batch_size
            corpus_data_np = np.load(self.test_corpus_path)
            self.current_epoch_size = self.get_epoch_size(corpus_data_np)
            print("TEST_FILE_PATH: %s, EPOCH_SIZE: %d" % (self.test_corpus_path, self.current_epoch_size))
            yield corpus_data_np, batch_words_num

    def __call__(self):
        self.status = self.status.strip().lower()
        if self.status == 'train':
            self.init_training_corpus()
            for corpus_np, batch_words_num in self.get_training_data():
                yield corpus_np, batch_words_num
        else:
            for corpus_np, batch_words_num in self.get_dev_test_data():
                yield corpus_np, batch_words_num


if __name__ == "__main__":
    '''
    Debug
    '''
    provide = ptb_data_provider()
    provide.status = 'train'
    for x, y, length in provide():
        print("input", x)
        print("output", y)
        print("length", length)
        input("Next")
