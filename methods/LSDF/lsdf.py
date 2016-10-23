from .conf import save_time
from .conf import timeit
from .conf import reset_lsdf_global_value
import utility.wrapper
from .conf import *


def compute_Sw(x_labeled, y_labeled, x_unlabeled, k = 5, gama = 100):
    """
    计算Sw矩阵，Sw矩阵反应了类内的联系。
    Sw(ij) = gama if node i and node j share the same label
    Sw(ij) = 1 if node i is KNN of node j or node j is KNN of node i.
    Sw(ij) = 0 otherwise.

    Input
    -----
    x_labeled: {numpy array}, shape {n_samples, n_features}, x_labeled,含有标签的数据，标签是y_labeled
    y_labeled: {numpy array}, shape {n_samples,}. y_labeled是标签。
    x_unlabeled: {numpy array}, shape {n_samples, n_features}, x_labeled是无标签的数据
    k: {int}, 是近邻数，default is 5
    gama: {float}, 类内边的权重，suggested 1~100

    Output
    ------
    Sw: {numpy array}, shape {n_samples, n_samples}

    notes
    -----
    对输出Sw矩阵做了约定：
        假设ln个带标签的样本
        假设un个未带标签的样本
        那么矩阵索引0~ln-1对应于带标签的样本，ln~ln+un-1对应于未带标签的样本
    """

    ln = x_labeled.shape[0]
    un = x_unlabeled.shape[0]
    x = np.concatenate((x_labeled, x_unlabeled), axis=0)
    n_samples = x.shape[0]
    S = pairwise_distances(x, metric='euclidean', n_jobs=1)
    """
    filter_KNN包括了自身，所以要k+1
    """
    s_knn = get_knn_flag(S, k=k+1)
    Sw = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if (i < ln and j < ln) and (y_labeled[i] == y_labeled[j]):
                Sw[i, j] = gama
            elif (i >= ln or j >= ln) and (s_knn[i, j] == True or s_knn[j, i] == True):
                Sw[i, j] = 1
            else:
                Sw[i, j] = 0
    return Sw


def compute_Sb(x_labeled, y_labeled, x_unlabeled):
    """
    计算Sb矩阵，Sb矩阵反应了类间的联系。

    Sb(ij) = 1 if node i and node j have different label
    Sw(ij) = 0 otherwise.

    Input
    -----
    x_labeled: {numpy array}, shape {n_samples, n_features}, x_labeled,含有标签的数据，标签是y_labeled
    y_labeled: {numpy array}, shape {n_samples,}. y_labeled是标签。
    x_unlabeled: {numpy array}, shape {n_samples, n_features}, x_labeled是无标签的数据

    Output
    ------
    Sb: {numpy array}, shape {n_samples, n_samples}


    对输出Sb矩阵做了约定：
        假设ln个带标签的样本
        假设un个未带标签的样本
        那么矩阵索引0~ln-1对应于带标签的样本，ln~ln+un-1对应于未带标签的样本
    """
    ln = len(x_labeled)
    ln = x_labeled.shape[0]
    un = x_unlabeled.shape[0]
    n_samples = ln + un
    Sb = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if (i < ln and j < ln) and y_labeled[i] != y_labeled[j]:
                Sb[i, j] = 1
    return Sb


def compute_Lw(Sw):
    """
    compute laplacian matrix according Sw

    $L_w = D_w - S_w$, 其中 $D_w = diag(S_w \mathbf{1})$

    Input
    -----
    Sw: {numpy array}, shape {n_samples, n_samples}

    Output
    ------
    Lw: {numpy array}, shape {n_samples, n_samples}
    """
    n_samples = Sw.shape[0]
    Dw = np.diag( np.dot(Sw, np.ones((n_samples, 1))).reshape(-1) )
    Lw = Dw - Sw
    return Lw


def compute_Lb(Sb):
    """
    compute laplacian matrix according Sb

    $L_b = D_b - S_b$, 其中 $D_b = diag(S_b \mathbf{1})$
    Input
    -----
    Sb: {numpy array}, shape {n_samples, n_samples}

    Output
    ------
    Lb: {numpy array}, shape {n_samples, n_samples}
    """
    n_samples = Sb.shape[0]
    Db = np.diag( np.dot(Sb, np.ones((n_samples, 1))).reshape(-1) )
    Lb = Db - Sb
    return Lb


def compute_Lr(x_labeled, x_unlabeled, Lw, Lb):
    """
    计算每个特征的得分
    $L_r = \frac{f_r^T L_b f_r}{f_r^T L_w f_r}$
    The score is larger, the feature is more important.
    其中：
    f_r 是所有样本的第r个特征值
    Input
    -----
    x_labeled: {numpy array}, shape {n_samples, n_features}, x_labeled,含有标签的数据，标签是y_labeled
    x_unlabeled: {numpy array}, shape {n_samples, n_features}, x_labeled是无标签的数据
    Lw: {numpy array}, shape {n_samples, n_features}, Lw is laplacian matrix of Sw
    Lb: {numpy array}, shape {n_samples, n_features}, Lb is laplacian matrix of Sb

    Output
    ------
    Lr: {numpy array}, shape {n_samples, n_features}, Lb is laplacian matrix of Sb
    """
    f_n = x_labeled.shape[1]
    Lr = np.zeros(f_n)
    x = np.concatenate((x_labeled, x_unlabeled))
    for f_i in range(f_n):
        x_fi = x[:, f_i:f_i+1]
        a = np.dot( np.dot( x_fi.T, Lb ), x_fi)
        b = np.dot( np.dot( x_fi.T, Lw ), x_fi)
        b = np.maximum(b, 10**-8)
        Lr[f_i] = a / b
    return Lr


@reset_lsdf_global_value
@timeit(utility.wrapper.LSDFTime)
def lsdf(x_labeled, y_labeled, x_unlabeled):
    """
    lsdf 算法，返回每个特征的得分。
    The lsdf score is larger, the feature is more important.

    核心公式：

    Sw
        Sw(ij) = gama if node i and node j share the same label
        Sw(ij) = 1 if node i is KNN of node j or node j is KNN of node i.
        Sw(ij) = 0 otherwise.

    Sb
        Sb(ij) = 1 if node i and node j have different label
        Sw(ij) = 0 otherwise.

    $L_w = D_w - S_w$, 其中 $D_w = diag(S_w \mathbf{1})$
    $L_b = D_b - S_b$, 其中 $D_b = diag(S_b \mathbf{1})$
    $L_r = \frac{f_r^T L_b f_r}{f_r^T L_w f_r}$,    其中 f_r 是所有样本的第r个特征值
    """
    f_n = x_labeled.shape[1]
    Sw = compute_Sw(x_labeled, y_labeled, x_unlabeled)
    Sb = compute_Sb(x_labeled, y_labeled, x_unlabeled)
    Lw = compute_Lw(Sw)
    Lb = compute_Lb(Sb)
    Lr = compute_Lr(x_labeled, x_unlabeled, Lw, Lb)

    return Lr


def feature_ranking(score):
    """
    Rank features in descending order according to lsdf score. The lsdf score is larger, the feature
    is more important.
    """
    feature_order = np.argsort(score)
    return feature_order[::-1]