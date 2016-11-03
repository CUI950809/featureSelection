#/usr/bin/python
# -*- coding=utf-8 -*-

from methods.LaplacianScore.conf import *

from .conf import timeit
from .conf import reset_LaplacianScore_global_value
import utility.wrapper
from .conf import *


def compute_S(x_train, k = 10, t = 100):
    """
    compute rbf distance for knn node.

    Input
    -----
    x_train: {numpy array}, shape {n_samples, n_features}.
    k: {int}. k is number of nearest neibours.
    t: {float}. gama in rbf. gama = 1 / t

    Output
    ------
    S_new: {numpy array}, shape {n_samples, n_samples}.
    """
    gama = 1.0/t
    # S = pairwise_kernels(x_train, metric='rbf', n_jobs=1)

    square_distance_m = pairwise_distances(x_train, metric='euclidean', squared=True , n_jobs=1)
    knn_flag_m = get_knn_flag(square_distance_m, k= k+1)
    # if node i is knn of node j or if node j is knn of node i. knn_flag_m[i][j] = 1
    knn_flag_m = knn_flag_m.T + knn_flag_m
    heat_similarity = np.exp(-gama * square_distance_m)
    heat_similarity[~knn_flag_m] = 0
    return heat_similarity


def compute_D(S):
    """
    计算D矩阵
    """
    n_sampels = S.shape[0]
    D = np.diag(np.dot( S, np.ones((n_sampels, 1)).reshape(-1) ))
    return D


def compute_L(D, S):
    """
    计算L矩阵
    """
    L = D - S
    return L


def compute_laplacian_score(x_train, L, D):
    """
    compute laplacian score for each feature

    Input
    -----
    x_train: {numpy array}, shape {n_samples, n_features}.
    L: {numpy array}, shape {n_samples, n_samples}.
    D: {numpy array}, shape {n_samples, n_samples}.

    Output
    ------
    feature_laplacian_score: {numpy array}, shape {n_samples, n_features}.
    """

    # 特征
    f_len = x_train.shape[1]
    x_f = set(range(f_len))

    n_sampels = x_train.shape[0]

    feature_laplacian_score = np.zeros(f_len)
    for f_i in x_f:
        # 第i个特征
        x_fi = x_train[:,f_i:f_i+1]

        # 求取新的fr
        a = np.dot( np.dot( x_fi.T, D ), np.ones((n_sampels, 1)) )
        b = np.dot( np.dot( np.ones((1, n_sampels)), D ), np.ones((n_sampels, 1)) )
        x_fi_new = x_fi - a / b * np.ones((n_sampels, 1))

        # 求L_i
        c = np.dot( np.dot( x_fi_new.T, L ), x_fi_new )
        d = np.dot( np.dot( x_fi_new.T, D ), x_fi_new )

        if d == 0:
            feature_laplacian_score[f_i] = float("inf")
        else:
            feature_laplacian_score[f_i] = c / d

    return feature_laplacian_score


@reset_LaplacianScore_global_value
@timeit(utility.wrapper.LaplacianScoreTime)
def laplacian_score(x_train, k= 10, t = 100):
    """
    compute laplacian score for each feature.
    For a good feature, Laplacian Score tends to be small.
    《Laplacian Score for Feature Selection》

    Input
    -----
    x_train: {numpy array}, shape {n_samples, n_features}.
    k: {int}. k is number of nearest neibours.
    t: {float}. gama in rbf. gama = 1 / t

    Output
    ------
    feature_score: {numpy array}, shape {n_features, }.
    """
    S = compute_S(x_train, k=k, t=t)
    D = compute_D(S)
    L = compute_L(D, S)
    feature_score = compute_laplacian_score(x_train, L, D)
    return feature_score


def feature_ranking(score):
    """
    For a good feature, Laplacian Score tends to be small.
    《Laplacian Score for Feature Selection》
    """
    feature_order = list( np.argsort(score) )
    return feature_order