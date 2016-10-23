from methods.SSelect.conf import nmi
from methods.SSelect.conf import np
from methods.SSelect.conf import pairwise_distances
from methods.SSelect.conf import get_knn_flag
from methods.SSelect.conf import *

from .conf import timeit
from .conf import reset_SSelect_global_value
import utility.wrapper


def compute_W_by_distM(square_distance, theta = 1):
    """
    change square distance matrix to guassion similarity matrix
    $ s_{ij} = e^{frac{-{||xi−xj||^2}}{2* {\theta}^2} $
    Input
    -----
    square_distance: {numpy array}, shape {n_samples, n_samples}
    theta: {float}.

    Output
    ------
    similarty: {numpy array}, shape {n_samples, n_samples}
    """
    similarty = np.exp(- square_distance / (2 * (theta ** 2 ) ) )
    return similarty


def compute_S(X, sum_w, d):
    """
    compute S

    $
    \large{\begin{array}{ll}
     \varphi (\mathbf{f}) = \mathbf{f} - \frac{\sum_i^n f_i d_i}{volV} \cdot \mathbf{1}
    \end{array}}

    volV is the sum of W
    $
    Input
    -----
    X: {numpy array}, shape {n_samples, n_features}
    W: {numpy array}, shape {n_samples, n_samples}
    d: {numpy array}, shape {n_samples,}

    Output
    ------
    S: {numpy array}, shape {n_samples, n_features}
    """

    X = X.copy()
    n_samples = X.shape[0]
    f_n = X.shape[1]
    S = np.zeros((n_samples, f_n))
    volv = sum_w
    for f_i in range(f_n):
        x_fi = X[:, f_i]
#         print(x_fi.shape)
        a = np.sum(x_fi*d)
        gi = x_fi - (a / volv) * np.ones(n_samples)
        S[:, f_i] = gi
    return S


def select_Sg(S, d):
    """
    compute Sg

    $
    newS = \{\mathbf{g}|\mathbf{g} \in \mathbb{R}^n,<\mathbf{g} \cdot \mathbf{d}>=0\}
    $

    Input
    -----
    S: {numpy array}, shape {n_samples, n_features}
    d: {numpy array}, shape {n_samples,}

    Output
    ------
    select_g: {list}.
    """
    fn = S.shape[1]
    select_g = []
    for i in range(fn):
        a = np.sum( S[:,i] * d )
        if a < 10 ** -9:
            select_g.append(i)
    return select_g


def compute_first_term(S, W, d, g_index, namuda = 0.1):
    """
    计算第一部分

    $ \lambda \frac{\sum_{v_i \tilde{} v_j}(g_i-g_j)^2 \times {\omega}_{ij}}
    {2\sum_{v_i \in V}\ g_i^2 \times d_i} $

    Input
    -----
    S: {numpy array}, shape {n_samples, n_features}
    W: {numpy array}, shape {n_samples, n_samples}
    d: {numpy array}, shape {n_samples,}
    g_index: {int}. feature index
    namuda: {float}.

    Output
    ------
    float value
    """
    n_samples = S.shape[0]
    numerator = 0.0
    denominator = 0.0
    for i in range(n_samples):
        gi = S[i, g_index]
        denominator += 2 * (gi ** 2) * d[i]
        for j in range(n_samples):
            gj = S[j, g_index]
            numerator +=  ( (gi - gj) ** 2 ) * W[i, j]

    return namuda * numerator / denominator


def compute_second_term(S, YL, g_index, namuda = 0.1):
    """
    计算分母部分

    $ (1-\lambda)(1-NMI(\tilde{\bf{g}},\bf{y})) $

    notes
    -----
    做一个约定，
    S矩阵的前面部分全部对应带标签的，
    后部分对应不带标签的

    Input
    -----
    S: {numpy array}, shape {n_samples, n_features}
    YL: {numpy array}, shape {n labeled samples,}
    g_index: {int}. feature index
    namuda: {float}.

    Output
    ------
    foat value
    """

    import sklearn as sk
    import time
    from sklearn import cluster

    label_true = YL.reshape(-1).copy()
    n_clusters = len( set(label_true) )
    ln = len(label_true)

    data = S[:ln, g_index:g_index+1]

    kmeans_model = cluster.KMeans(n_clusters=n_clusters, random_state=int( time.clock() ))
    label_pred = kmeans_model.fit_predict(data)
    nmi_value = nmi(label_true, label_pred)
#     print(nmi_value)
#     return nmi_value
    return (1.0-namuda) * ( 1.0 - nmi_value)


@reset_SSelect_global_value
@timeit(utility.wrapper.SSelectTime)
def SSelect(XL, YL, XU, k = 10, theta = 1, namuda = 0.1):
    """
    compute Lr score for each feature.

    the score is bigger, the more relevance features it is. so sort in descending order
    can get the top k most relevance features.

    reference:
    Zheng Zhao, Huan Liu. "Semi-supervised Feature Selection via Spectral Ananlysis"

    Input
    -----
    XL: {numpy array}, shape {n labeled samples, n_features}
    YL: {numpy array}, shape {n labeled samples}
    XU: {numpy array}, shape {n unlabeled samples, n_features}
    k: {int}. k nearest neigbours. default 10
    theta: parameter in guassion similarity. default 10.
           $ s_{ij} = e^{frac{-{||xi−xj||^2}}{2* {\theta}^2} $
    namuda: default 0.1.

    Output
    ------
    Lr: {numpy array}, shape{n_features,}. it is all features score.
    """

    X = np.concatenate((XL, XU), axis = 0)
    YL = YL.copy()

    distance_array = pairwise_distances(X, metric="euclidean", squared=True, n_jobs=1)
    knn_flag_array = get_knn_flag(distance_array, k=k)

    W = compute_W_by_distM(distance_array, theta=theta)
    knn_flag_array = knn_flag_array + knn_flag_array.T
    # 非k近邻元素设置为0
    W[~knn_flag_array] = 0

    # 求每行的权值的和
    d = np.sum(W, axis=1)
    sum_w = np.sum(W)
    S = compute_S(X, sum_w, d)

#     print(W.shape, d.shape)

    gn = S.shape[1]
    Lr = np.zeros(gn)
    for g_index in range(gn):
        first_term = compute_first_term(S, W, d, g_index, namuda= namuda)
        second_term = compute_second_term(S, YL, g_index, namuda= namuda)
#         print(first_term, second_term)
        Lr[g_index] = first_term + second_term
    return Lr


def feature_ranking(score):
    """
    the score is bigger, the more relevance features it is. so sort in descending order
    can get the top k most relevance features.
    """
    feature_order = np.argsort(score)
    return feature_order[::-1]




