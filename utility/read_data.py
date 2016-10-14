#!/usr/bin/python
# -*- coding:utf-8 -*-

from utility.conf import np
from utility.conf import pd
from utility.conf import warnings
from utility.conf import *


def read_feature(path, sep=",", header=None, engine="python"):
    data = pd.read_csv(path, sep=sep, header=header, engine=engine)
    data_values = data.values
    return data_values


def read_label(path, sep=",", header=None, engine="python"):
    """
    读入二维 n x 1 数据
    转换为一维 n 数据
    """
    data = pd.read_csv(path, sep=sep, header=header, engine=engine)
    data_values = data.values
    data_d1 = np.reshape(data_values, (1,-1))[0]
    return data_d1


def label_n1_to_nc(label):
    """
    一维 n => 二维 n x c
    """
    c = int(np.max(label)) + 1
    n = len(label)
    tmp_label = np.zeros((n, c))
    for i,j in enumerate(label):
        tmp_label[int(i),int(j)] = 1
    label = tmp_label
    return label


def label_nc_to_n1(label):
    """
    一维 n => 二维 n x c
    """
    n = len(label)
    tmp_label = np.zeros(n)
    for i, arr in enumerate(label):
        for j, v in enumerate(arr):
            if v > 0:
                tmp_label[i] = j
    label = tmp_label
    return label


def read_data(paths, fdname, lname, sep=",| |\t", header=None):
    """
    Input
    -----
    paths: {list}. paths consist many folder paths which include many data folders.
    fdname, {str}. features data file name. the data in it has shape {n_samples, n_features}
    lname, {str}. label file name. the label in it has shape {n_samples,}

    Output(yield)
    ------
    feature: {numpy array}, shape {n_samples, n_features}
    cluster_names: {numpy array}, shape {n_samples,}
    """
    paths = paths
    data_file_name = fdname
    label_file_name = lname

    for file_path in paths:
        file_path = file_path.strip()
        feature = read_feature(file_path + '/' + data_file_name, sep=sep, header=header, engine="python")
        cluster_names = read_label(file_path + '/' + label_file_name, sep=sep, header=header, engine="python")

        yield feature, cluster_names


def selected_data_by_flag(x, y, test_flag, fth):
    """
    yield train data and test data from x, y
    Input
    -----
    x: {numpy array}, shape {n_samples, n_features}.
    y: {numpy array}, shape {n_samples,}.
    test_bool: {numpy array}, shape {n_samples}.
    fth: {list}.

    Output
    ------
    yield x_train, y_train, x_test, y_test
    """
    if np.max(fth) > np.max(test_flag):
        raise warnings.warn('max value in fth bigger than test_flag!')

    if x.shape[0] != test_flag.shape[0] or y.shape[0] != test_flag.shape[0]:
        raise ValueError('test_flag lengths is not eaqual to n_samples')

    if x.shape[0] != y.shape[0]:
        raise ValueError('samples number not eaqual to y lengths')

    for v in fth:
        x_train = x[test_flag > v]
        y_train = y[test_flag > v]
        x_test = x[test_flag <= v]
        y_test = y[test_flag <= v]
        yield x_train, y_train, x_test, y_test
