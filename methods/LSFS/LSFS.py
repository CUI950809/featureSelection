#!usr/bin/python
# -*- coding:utf-8 -*-

from methods.LSFS.conf import *

import utility.wrapper
from utility.wrapper import timeit
from utility.wrapper import get_object_value
from utility.wrapper import reset_lsfs_global_value


LSFS_NITER = 6
W_NITER = 10


def value_variation(cur_f, pre_f):
    b = 10**-8
    if cur_f != 0:
        b = cur_f
    return abs((cur_f - pre_f) / b)


def norm_2_1_nz(a):
    """
    compute norm L2_1 for a matrix
    $\sum_j\sqrt{(\mathbf{w}^j)^T \mathbf{w}^j + \varepsilon}$

    Input
    -----
    a: {numpy array}, shape {}
    """
    return np.sum( np.sqrt( np.linalg.norm(a, ord = 2, axis=1) ** 2 + 10**-8 ) )


def compute_Q(W):
    """
    $ \large q_{jj} = \frac
                        {\sum_j\sqrt{(\mathbf{w}^j)^T \mathbf{w}^j + \varepsilon}}
                        {\sqrt{(\mathbf{w}^j)^T\mathbf{w}^j + \varepsilon}}$

    其中，\mathbf{w}^j 表示 第w的第j行

    axis = 1 =》 对W的每一行求L2范数
    np.linalg.norm(W, ord=2, axis = 1) =》 对每行求L2范数
    """
    Q = norm_2_1(W) / ( np.linalg.norm(W, ord = 2, axis=1) + 10**-6 )
    Q = np.diag(Q)
    return Q


def norm_2_1(a):
    """
    ${||M||}_{2,1} = \sum_{i=1}^n \sqrt{\sum_{j=1}^m m_{ij}^2}$
    """
    return np.sum(np.linalg.norm(a, ord = 2, axis=1))


def fun22_value(W, X, H, Q, Y, gama):
    """
    ||H*X.T*W - H*Y||_{fro}^ 2 + gama * trac(W.T*Q*W)
    """
#    gama = 10**-6
    return np.linalg.norm(np.dot(np.dot(H, X.T), W) - np.dot(H, Y), ord = "fro")**2 + gama*np.trace(np.dot(np.dot(W.T, Q),W))


@get_object_value(utility.wrapper.LSFSWObejectV)
def fun17_value(W, X, Y, gama):
    """
    $
    \min_\textbf{W}({||\textbf{X}^T\textbf{W} + \textbf{1}\textbf{b}^T - \textbf{Y}||}^2_F 
          + \gamma{||\textbf{W}||^2_{2,1}} )
    $
    """
#    gama = 10**-6
#     print(X.shape, Y.shape, W.shape)
    b = compute_b(X, Y, W)
#     print("17 value", np.linalg.norm(np.dot(X.T, W) + np.dot(np.ones((X.shape[1],1)), b.T) - Y, ord = "fro")**2 + gama*(norm_2_1(W)**2))
    return np.linalg.norm(np.dot(X.T, W) + np.dot(np.ones((X.shape[1],1)), b.T) - Y, ord = "fro")**2 + gama*(norm_2_1(W)**2)


@get_object_value(utility.wrapper.LSFSObejectV)
def fun8_value(X, Y, W, b, gama):
    """
    X : d x n

    ||X.T * W + 1*b.T - Y||_2^2 + gama * ( ||W||_{Fro} ^ 2 )
    """
#    gama = 10**-6
    n = X.shape[1]

    return np.linalg.norm( np.dot(X.T,W) + np.dot(np.ones((n, 1)),b.T) - Y , ord=2)**2 + gama*(np.linalg.norm( W , ord = "fro")**2)


@timeit(utility.wrapper.LSFSCWTime)
def compute_W(X, Y, H, Q, gama):
    """
     W = (X*H*X.T + gama * Q)^-1 * X*H*Y

    Input
    -----
    X: {numpy array}, shape {n_features, n_samples}
    Y: {numpy array}, shape {n_samples,}
    H: {numpy array}, shape {n_samples, n_samples}
    gama: {float}, regular value

    Output
    ------
    W: {numpy array}, shape {n_samples, n_samples}

    """
    W = np.dot( np.dot( np.dot( \
                np.linalg.inv( np.dot( np.dot(X,H), X.T)+gama*Q ) \
                               , X), H), Y)
    return W


def compute_H(n):
    """
    I: shape {n, n}
    1: shape {n,}
    H = I - 1/n * 1 * 1.T

    Input
    -----
    n: {int}

    Output
    ------
    H: {numpy array}, shape {n, n}
    """
    H = np.eye(n,n) - 1/n*np.ones((n,n))
    return H


@timeit(utility.wrapper.LSFSGetWTime)
def get_W(X, Y, gama):
    """
    compute W

    $\textbf{W}_{t+1} = (\textbf{XHX^T} + \gamma \textbf{Q})^{-1}\textbf{XHY}$

    Input
    -----
    X: {numpy array}, shape {n_features, n_samples}
    Y: {numpy array}, shape {n_samples,}
    gama: regular value

    Output
    ------
    W: {numpy array}, shape {n_samples, n_samples}
    """

    global W_NITER

    d, n = X.shape
    c = Y.shape[1]

    # Q初始化为一个单位矩阵
    Q = np.eye(d)

#     print(Q)
#     print("====================")

    # H矩阵不变，算一遍即可
    H = compute_H(n)


    W = compute_W(X, Y, H, Q, gama )

    Q = compute_Q(W)
#    pre_f = cur_f = fun22_value(W, X, H, Q, Y, gama = gama)
    pre_f = cur_f = fun17_value(W, X, Y, gama)

    # nan 判断
    if cur_f != cur_f:
        print("{0} nan".format(__file__))
        return W

    NITER = W_NITER
    #NITER = 50

    epsilon = 10**-8
    for i in range(NITER):
        pre_f = cur_f
        W = compute_W(X, Y, H, Q, gama)
        Q = compute_Q(W)

        cur_f = fun17_value(W, X, Y, gama)

        # nan 判断
        if cur_f != cur_f:
            print("{0} nan".format(__file__))
            break

        # coverage
        if value_variation(cur_f, pre_f) < epsilon:
            break

    return W


def compute_YU(X, W, b):
    """
    compute YU

        min   ( ||(x_i.T) * W + b.T - yi.T||_{Fro} ^ 2 )
    s.t. yi>=0, 1*yi=1

    Input
    -----
    X: {numpy array}, shape {n_features, n_samples}
    W: {numpy array}, shape {n_features, n_clusters}
    W: {numpy array}, shape {n_clusters, 1}

    Output
    ------
    YU: {numpy array}, shape {n_samples, n_clusters}

    """
    c = W.shape[1]
    YU = np.zeros((X.shape[1], c))
    # 对于每一个样本，维度 1 x d
    for i in range(X.shape[1]):
        """
        min ( ||(xi.T) * W + b.T - yi.T||_{Fro} ^ 2 )
        s.t. yi>=0, 1*yi=1
        """
        ad = np.dot(X[:,i:i+1].T, W) + b.T
        ad_new, ft = EProjSimplex_new(ad)
        YU[i:i+1,:] = ad_new.A
    return YU


def compute_b(X, Y, W):
    """
    b = 1/n * (Y.T * 1 - W.T * X * 1)
    1 是 n x 1 维的全1矩阵

    Input
    -----
    X: {numpy array}, shape {n_features, n_samples}
    Y: {numpy array}, shape {n_samples, n_clusters}
    W: {numpy array}, shape {n_features, n_clusters}

    Output
    ------
    b: {numpy array}, shape {n_clusters,}
    """
    n = X.shape[1]
    b = 1/n*(np.dot(Y.T, np.ones((n,1))) - np.dot(np.dot(W.T, X), np.ones((n,1))))
    return b


def get_X(XL, XU=None):
    """
    Input
    -----
    XL: {numpy array}, shape {n_features, n1 labeled samples}
    XU: {numpy array}, shape {n_features, n2 unlabeled samples}

    Output
    -----
    X: {numpy array}, shape {n_features, n1 + n2 samples}
    """
    if XU is None:
        return XL

    X = sp.concatenate((XL, XU), axis=1)
    return X


def get_Y(YL, YU=None):
    """
        Input
        -----
        YL: {numpy array}, shape {n1 labeled samples, n_clusters}
        YU: {numpy array}, shape {n2 unlabeled samples, n_clusters}

        Output
        -----
        Y: {numpy array}, shape {n1 + n2 samples, n_clusters}
        """
    if YU is None:
        return YL

    Y = sp.concatenate((YL,YU), axis=0)
    return Y


def get_new_W_b(XL, YL, XU,  gama, YU=None):
    """
    Input
    -----
    XL: {numpy array}, shape {n_features, n labeled samples}
    YL: {numpy array}, shape {n labeled samples, n_clusters}
    XU: {numpy array}, shape {n_features, n unlabeled samples}
    gama: {float}, regular value
    YU: {numpy array}, shape {n unlabeled samples, n_clusters}. default None

    Output
    ------
    W: {numpy array}, shape {n_features, n_clusters}
    b: {numpy array}, shape {n_clusters,}
    """
#     n = X.shape[1]

    cw_X = XL
    cw_Y = YL

    # 第一次调用的时候, YU仍不知道
    if YU is not None:
        cw_X = get_X(XL, XU)
        cw_Y = get_Y(YL, YU)


    W = get_W(cw_X, cw_Y, gama)

#     print_W(W)
#     b = 1/n*(np.dot(Y.T, np.ones((n,1))) - np.dot(np.dot(W.T, X), np.ones((n,1))))

    b = compute_b(cw_X, cw_Y, W)

    return W, b


def compute_thea(W):

    """
    W : d x c
    thea_j = ||w_j||2 / sum(||w_j||2)
                        j=1:d
    Input
    -----
    W: {numpy array}, shape {n_features, n_clusters}

    Output
    ------
    s: {numpy array}, shape {n_features,}
    """
    # 对W的每行求L2范数，再求和
    W_L2_sum = np.sum(np.linalg.norm(W, ord=2, axis = 1))
    # 对W的每行求L2范数
    s = np.linalg.norm(W, ord=2, axis = 1) / W_L2_sum
    return s


@reset_lsfs_global_value
@timeit(utility.wrapper.LSFSTime)
def LSFS(XL, YL, XU, gama = 10**-1):
    """
    Input
    -----
    XL: {numpy array}, shape {n_features, n labeled samples}
    YL: {numpy array}, shape {n labeled samples, n_clusters}
    XU: {numpy array}, shape {n_features, n unlabeled samples}
    gama: {float}, regular value

    notes
    -----
    if bigger score the features get, the more importance features it is.

    Output
    ------
    s: {numpy array}, shape {n_features,}

    """

    global LSFS_NITER

    W, b = get_new_W_b(XL, YL, XU, gama)
    YU = compute_YU(XU, W, b)
    new_X = get_X(XL, XU)
    new_Y = get_Y(YL, YU)
    cur_f = fun8_value(new_X, new_Y, W, b, gama)

    # nan 判断
    if cur_f != cur_f:
        s = compute_thea(W)
        feature_order = list( np.argsort(s) )
        feature_order = feature_order[::-1]
        print("{0} wai nan".format(__file__))
        return feature_order, None

    NITER = LSFS_NITER
    # NITER = 10
    epsilon = 10**-8
    for i in range(NITER):
        pre_f = cur_f

        W, b = get_new_W_b(XL, YL, XU, gama, YU=YU)
        YU = compute_YU(XU, W, b)
        new_X = get_X(XL, XU)
        new_Y = get_Y(YL, YU)
        cur_f = fun8_value(new_X, new_Y, W, b, gama)

        # nan 判断
        if cur_f != cur_f:
            print("{0} nei nan".format(__file__))

        if value_variation(cur_f, pre_f) < epsilon:
            break

    s = compute_thea(W)
    return s


def feature_ranking(score):
    """
    Rank features in descending order according to lsfs score. The lsfs score is larger, the feature
    is more important.
    """
    feature_order = np.argsort(score)
    return feature_order[::-1]

