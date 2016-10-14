#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def fun0(x, b):
    """ y = 3"""
    return b*np.ones(len(x))

def fun1(x, b):
    """y = -x + b"""
    return -x + b

def shuffle_samples(x, y):
    """
    shuffles samples

    Input
    -----
    x: {numpy array}, shape {n_samples, n_features}
    y: {numpy array}, shape {n_samples,}

    Output
    ------
    new_x: {numpy array}, shape {n_samples, n_features}
    new_y: {numpy array}, shape {n_samples,}
    """
    n_samples = len(x)
    index = list(range(n_samples))
    np.random.shuffle(index)
    new_x = x[index, :]
    new_y = y[index]
    return new_x, new_y

def gen_data(n_samples = 1000):
    """
    random gen data, which consist of two class data.
    one is according to 'y = 3'
    anather is according to 'y = -x + b'

    Input
    -----
    n_samples: {int}, the numbers of samples

    notes
    -----
    n_samples = n_samples1 + n_samples2

    Output
    ------
    x_train: {numpy array}, shape {n_samples1, n_features}
    x_test: {numpy array}, shape {n_samples2, n_features}
    y_train: {numpy array}, shape {n_samples1,}
    y_test: {numpy array}, shape {n_samples2,}
    """
    x = None
    y = None

    np.random.seed(13)

    c1_num = int(n_samples/2)
    c2_num = n_samples - int(n_samples/2)

    b1 = 3
    c1_x1 = np.random.uniform(b1, b1+1, size = c1_num)
    c1_x1.sort()
    c1_x2 = fun0(c1_x1, b1)
    y1 = np.zeros(c1_num) + 0

    b2 = 10
    c2_x1 = np.random.uniform(5, b2, size = c2_num)
    c2_x1.sort()
    c2_x2 = fun1(c2_x1, b2)
    y2 = np.zeros(c2_num) + 1

    c1_x = np.concatenate((c1_x1[:, np.newaxis], c1_x2[:, np.newaxis]), axis = 1)
    c2_x = np.concatenate((c2_x1[:, np.newaxis], c2_x2[:, np.newaxis]), axis = 1)
#     c2_x = np.concatenate((c2_x1[np.newaxis, :], c2_x2[np.newaxis, :]))

    print(c1_x.shape)
    print(c2_x.shape)

    x = np.concatenate((c1_x, c2_x), axis=0)

    # 在原始数据基础上加入噪音
    x[:c1_num,1:2] = x[:c1_num,1:2] + 3*np.random.normal(size=c1_num)[:, np.newaxis]
    x[c1_num:n_samples,1:2] = x[c1_num:n_samples,1:2] + 0.5*np.random.normal(size=c2_num)[:, np.newaxis]

    print(x.shape)

    y = np.concatenate((y1, y2))

    x, y = shuffle_samples(x, y)

    # 随机生成一个bool数组，筛选训练数据
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train = x[train_mask, :], y[train_mask]
    x_test, y_test = x[~train_mask, :], y[~train_mask]
    return x_train, x_test, y_train, y_test


def plot_data(x_train, x_test, y_train, y_test, figsize=(8,5)):
    """
    plot samples data
    """
    fig = plt.figure(figsize=figsize)

    plt.scatter(x_train[y_train == 0,0], x_train[y_train == 0,1], color = "blue", label='train data class1')
    plt.scatter(x_train[y_train == 1,0], x_train[y_train == 1,1], color = "yellow", label='train data class2')
    plt.scatter(x_test[y_test == 0,0], x_test[y_test == 0,1], color = "red", label='test data class1')
    plt.scatter(x_test[y_test == 1,0], x_test[y_test == 1,1], color = "green", label='test data class2')

    plt.legend(loc='best')

    plt.xlim((0, 10))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()