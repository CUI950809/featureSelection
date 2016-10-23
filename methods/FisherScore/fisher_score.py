from .conf import *

from .conf import timeit
from .conf import reset_FisherScore_global_value
import utility.wrapper


@reset_FisherScore_global_value
@timeit(utility.wrapper.FisherScoreTime)
def fisher_score(X, y):
    """
    compute fisher scores of each feature.
    $F_r = \sum_{i=1}^c n_i*({\mu}_i - {\mu})^2  /  \sum_{i=1}^c n_i * {\theta}_i^2$

    note
    ----
    the fisher score is bigger,the feature is more important

    Input
    -----
    X: {numpy array}, shape {n_samples, n_features}
    y: {numpy array}, shape {n_samples,}

    Output
    ------
    score: {numpy array}, shape {n_features,}
    """

    # 特征
    f_len = X.shape[1]
    x_f = set(range(f_len))

    # 类别
    y_c = set(y)

    score = np.zeros(f_len, dtype=np.float)

    # 对于每个特征
    for f_i in x_f:
        # i特征的数据
        x_fi = X[:, f_i:f_i+1]

        """
        i特征的均值和方差
        """
        fi_mean = np.mean(x_fi)

        # 分子
        sum_1 = 0.0
        # 分母
        sum_2 = 0.0
        # 对于每个特征的每个类别
        for c_j in y_c:
            y_cj = y == c_j
            n_fi_cj = np.sum(np.array(y_cj))
            # i特征j类别的均值和方差
            fi_cj_mean = np.mean(x_fi[y_cj, :], axis=0)
            fi_cj_var = np.var(x_fi[y_cj, :], axis=0)

            sum_1 += n_fi_cj*(sum(fi_cj_mean - fi_mean)**2)
            sum_2 += n_fi_cj*(fi_cj_var**2)

        # pretect sum_2 from zeros
        if sum_2 < 1e-12:
            sum_2 = 1e-6

        score[f_i] = sum_1/sum_2
    return score


def feature_ranking(score):
    """
    Rank features in descending order according to fisher score. The fisher score is larger, the feature
    is more important.
    """
    feature_order = np.argsort(score)
    return feature_order[::-1]
