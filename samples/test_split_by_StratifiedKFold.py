from samples.conf import split_by_StratifiedKFold
from samples.conf import *

def test_split_by_StratifiedKFold():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
              0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

    n_folds = 10
    test_flag = split_by_StratifiedKFold(y, n_folds = n_folds)

    for i in range(n_folds):
        ith = i + 1
        assert np.sum(y == 0) * ith/n_folds - np.sum(y[test_flag <= i] == 0) < 1
        assert np.sum(y == 1) * ith/n_folds - np.sum(y[test_flag <= i] == 1) < 1
        assert np.sum(y == 0) * (n_folds - ith) / n_folds - np.sum(y[(test_flag >= i)] == 0) < 1
        assert np.sum(y == 1) * (n_folds - ith) / n_folds - np.sum(y[(test_flag) >= i] == 1) < 1

    print('test_split_by_StratifiedKFold successfully!')


if __name__ == "__main__":
    test_split_by_StratifiedKFold()