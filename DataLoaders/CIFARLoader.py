import random
import scipy
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import h5py


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.batch_size = cfg.batch_size
        self.data_path = '/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/CIFAR-10/cifar-10_96.h5'
        h5f = h5py.File(self.data_path, 'r')
        x_train = h5f['x_train'][:]
        y_train = h5f['y_train'][:]
        h5f.close()
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)
        self.x_train, self.y_train = self.preprocess(x_train, y_train, one_hot=cfg.one_hot)
        self.num_tr = x_train.shape[0]

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            index = np.random.randint(0, self.num_tr, size=self.cfg.batch_size)
            x = self.x_train[index]
            y = self.y_train[index]
        elif mode == 'valid':
            x = self.x_valid[start:end]
            y = self.y_valid[start:end]
        return x, y

    def get_validation(self):
        h5f = h5py.File(self.data_path, 'r')
        x_valid = h5f['x_test'][:]
        y_valid = h5f['y_test'][:]
        h5f.close()
        self.x_valid, self.y_valid = self.preprocess(x_valid, y_valid, one_hot=self.cfg.one_hot)

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation]

    def preprocess(self, x, y, normalize='standard', one_hot=True):
        if normalize == 'standard':
            x = (x - self.mean) / self.std
        elif normalize == 'unity_based':
            x /= 4096.
        x = x.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.channel)).astype(np.float32)
        if one_hot:
            y = (np.arange(self.cfg.num_cls) == y[:, None]).astype(np.float32)
        return x, y
