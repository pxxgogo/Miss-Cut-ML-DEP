from __future__ import print_function

import json
import os
import random
import sys
import traceback
from os import path as op
import cProfile
from io import StringIO
import pstats
# from memory_profiler import profile

import numpy as np

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

        self._parse_config()
        self._read_data()

    def _parse_config(self):
        try:
            assert op.isfile(CONFIG_FILE_NAME) and os.access(CONFIG_FILE_NAME, os.R_OK)
            self.config = json.load(open(CONFIG_FILE_NAME, 'r'))
            assert 'data_dir' in self.config
            self.data_dir = self.config['data_dir']
            assert op.isdir(self.data_dir) and os.access(self.data_dir, os.R_OK)
            self.filenames = os.listdir(self.data_dir)
            self.corpus_paths = []
            for filename in self.filenames:
                if not filename.endswith("dat.npy"):
                    continue
                fullname = op.join(self.data_dir, filename)
                print(fullname)
                assert op.isfile(fullname) and os.access(fullname, os.R_OK)
                self.corpus_paths.append(fullname)
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


    def decoder(self, file_path):
        ret_list = np.fromfile(file_path, dtype=np.int16, sep=" ")
        print("total words:", ret_list.shape[0])
        index_list = np.fromfile(file_path + '.index', dtype=np.int64, sep=" ")
        print("total lines:", index_list.shape[0])
        return ret_list, index_list

    def decoder_np(self, file_path, batch_flag=True):
        ret = np.load(file_path)
        return ret

    def _read_data(self):
        train_path = op.join(self.data_dir, self.filenames[0])
        valid_path = op.join(self.data_dir, self.filenames[1])
        test_path = op.join(self.data_dir, self.filenames[2])
        vocab_path = op.join(self.data_dir, self.filenames[3])
        self.vocab = json.load(open(vocab_path, 'r'))
        self.vocab["<eos>"] = len(self.vocab) + 1
        self.vocab_nums = self.vocab.values()

        # read from normal txt files
        self.training_data, self.training_index_data = self.decoder(train_path)
        print("finish reading training_data")

        self.valid_data, self.valid_index_data = self.decoder(valid_path)

        self.test_data, self.test_index_data = self.decoder(test_path)

    def get_config(self):
        return self.model_config

    def get_epoch_size(self):
        if self.status == 'train':
            return (self.training_index_data.shape[0]) // self.batch_size - 1
        elif self.status == 'valid':
            return (self.valid_index_data.shape[0]) // self.batch_size - 1
        elif self.status == 'test':
            return self.test_index_data.shape[0] - 1
        else:
            return None

    def get_ans_data(self, No, type):
        input = []
        output = []
        length = []
        if type == 0:
            data = self.training_data
            if No == 0:
                min_index = 0
            else:
                min_index = self.training_index_data[No * self.batch_size - 1] + 1
            max_index = self.training_index_data[(No + 1) * self.batch_size]
            indexes = self.training_index_data[No * self.batch_size: (No + 1) * self.batch_size]
        elif type == 1:
            data = self.valid_data
            if No == 0:
                min_index = 0
            else:
                min_index = self.valid_index_data[No * self.batch_size - 1] + 1
            max_index = self.valid_index_data[(No + 1) * self.batch_size]
            indexes = self.valid_index_data[No * self.batch_size: (No + 1) * self.batch_size]
        elif type == 2:
            data = self.test_data
            if No == 0:
                min_index = 0
            else:
                min_index = self.test_index_data[No - 1] + 1
            max_index = self.test_index_data[No + 1]
            indexes = self.test_index_data[No: No + 1]
        else:
            raise False
        data = data[min_index: max_index + 1]
        # print(min_index, max_index, type)
        # print(data)
        feed_data_list = []
        old_index = min_index
        max_length = -1
        for index in indexes:
            line = data[old_index - min_index: index + 1 - min_index]
            # print(line.shape)
            feed_data_list.append(line)
            if index - old_index > max_length:
                max_length = index - old_index
            old_index = index + 1
        # print(feed_data_list)
        for line in feed_data_list:
            # print(line)
            if line.shape[0] == 0:
                print(feed_data_list)
                print(line)
            line_length = line.shape[0] - 1
            input_np_array = np.zeros([max_length])
            output_np_array = np.zeros([max_length])
            output_np_array[:line_length] = line[1:]
            input_np_array[:line_length] = line[:-1]
            output.append(output_np_array)
            input.append(input_np_array)
            length.append(line_length)
        return input, output, length, max_length

    def __call__(self):
        self.status = self.status.strip().lower()
        epoch_size = self.get_epoch_size()
        print("epoch_size", epoch_size)
        if self.status == 'train':
            for i in range(epoch_size):
                data = self.training_data[i * self.batch_size: (i + 1) * self.batch_size]
                x, y, length, max_length = self.get_ans_data(i, 0)
                if max_length > 100:
                    print("final epoch_size", epoch_size)
                    break
                # print("input", x)
                # print("output", y)
                yield (x, y, length)
        elif self.status == 'valid':
            # self.yield_pos[1] = (self.yield_pos[1] + 1) % self.valid_data.shape[1]
            # i = self.yield_pos[1]
            for i in range(epoch_size):
                data = self.valid_data[i * self.batch_size: (i + 1) * self.batch_size]
                x, y, length, max_length = self.get_ans_data(i, 1)
                if max_length > 100:
                    print("final epoch_size", epoch_size)
                    break
                # print("input", x)
                # print("output", y)
                yield (x, y, length)
        else:
            # self.yield_pos[2] = (self.yield_pos[2] + 1) % self.test_data.shape[0]
            # i = self.yield_pos[2]
            for i in range(epoch_size):
                data = self.test_data[i: (i + 1)]
                x, y, length, max_length = self.get_ans_data(i, 2)
                if max_length > 100:
                    print("final epoch_size", epoch_size)
                    break
                # print("input", x)
                # print("output", y)
                yield (x, y, length)


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
