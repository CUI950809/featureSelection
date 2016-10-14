#!usr/bin/python
# -*- coding:utf-8 *-

from methods.PRPC.conf import stats
from methods.PRPC.conf import np
from methods.PRPC.conf import *

def PRPC(XL, YL, XU, num = 100):
    """
    order the feature by correlation coefficient.

    $$
    F_k = argmax_{Fj \in \mathbf{F}_a}[P(F_j, Y_L) - 
            \frac{1}{k-1}\sum_{F_i \in \mathbf{F}_s}P(F_j, F_i)]
    $$

    Input
    -----
    XL: {numpy array}, shape {n labeled samples, n_features}
    YL: {numpy array}, shape {n labeled samples}
    XU: {numpy array}, shape {n unlabeled samples, n_features}

    Output
    ------
    features_order: {numpy array}, shape {n_features,}
    """

    # 特征的索引号
    F = set(range(XL.shape[1]))
    Fs = set()
    features_order = []
    Fa = F - Fs

    ln = XL.shape[0]
    X = sp.concatenate((XL, XU), axis = 0)

    # YL, 坐标当作 -1
    pdata = dict()

    n = 0
    a = 0

    for k in range(1, num + 1):

        max_pearson = float("-inf")
        Fk = -1

        print(k)

        for Fj in Fa:

            pearson1 = 0
            if (Fj, -1) in pdata:
                pearson1 = pdata[(Fj, -1)]
                n += 1
            else:
                pdata[(Fj, -1)] = pearson1 = stats.pearsonr(X[:ln,Fj], YL)[0]

            pearson2_sum = 0

            a += 1
            for Fi in Fs:
                a += 1
                pearson2 = 0
                if (Fj, Fi) in pdata:
                    pearson2 = pdata[(Fj, Fi)]
                    n += 1
                else:
                    pdata[(Fj, Fi)] = pearson2 = stats.pearsonr(X[:,Fj], X[:,Fi])[0]
                pearson2_sum += pearson2

            if k - 1 == 0:
                pearson2_sum = 0
            elif k > 1:
                pearson2_sum /= k - 1

            pearson1 -= pearson2_sum

            if pearson1 > max_pearson:
                max_pearson = pearson1
                Fk = Fj

        Fs = Fs | {Fk}
        Fa = Fa - {Fk}

        features_order.append(Fk)

    print('pearson1 compute rate : {0}'.format(n/a))

    return features_order

