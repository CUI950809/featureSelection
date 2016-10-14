from utility.conf import cross_validation
from utility.conf import *


def split_by_StratifiedKFold(y,  n_folds = 3):
    """
    split test data to 1/n_folds, 2/n_folds,...,(n_folds-1)/n_folds of n_samples.
    other data, (n_folds-1)/n_folds, ..., 2/n_folds, 1/n_folds, as train data.

    split keep label balance.

    Input
    -----
    y: {numpy array}, shape {n_samples,}
    n_folds: {int}

    Output
    ------
    test_flag: {numpy array}, shape {n_samples,}
    """
    ss = cross_validation.StratifiedKFold(y, n_folds = n_folds, shuffle = True)
    n_samples = y.shape[0]
    test_flag = np.zeros(n_samples, dtype=np.int)
    for fold_idx, (_, test_idx) in enumerate(ss):
        test_flag[test_idx] = fold_idx
    return test_flag




