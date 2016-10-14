from samples.conf import kmax
from samples.conf import np
from samples.conf import *


def test_kth():
    a = np.array([1, 4, 3, 1, 3, 3, 1])
    # print(len(a.reshape(-1)))
    assert kmax.k_th(a, 0, len(a) - 1, 3) == 1
    assert kmax.k_th(a, 0, len(a) - 1, 4) == 3
    assert kmax.k_th(a, 0, len(a) - 1, 7) == 4
    print("k_th() successful!")


def test_get_knn_flag():
    a = np.array([[1, 4, 3, 1, 3, 3, 1]])
    bincount3 = np.array([0, 3, 0, 0, 0])
    bincount4 = np.array([0, 3, 0, 1, 0])
    bincount7 = np.array([0, 3, 0, 3, 1])

    result3 = kmax.get_knn_flag(a, 3)
    result4 = kmax.get_knn_flag(a, 4)
    result7 = kmax.get_knn_flag(a, 7)

    assert 0 == np.sum(bincount3 != np.bincount(a[result3], minlength=len(bincount3)))
    assert 0 == np.sum(bincount4 != np.bincount(a[result4], minlength=len(bincount4)))
    assert 0 == np.sum(bincount7 != np.bincount(a[result7], minlength=len(bincount7)))

    print("get_knn_flag() successful!")


def test_partition():
    a = np.array([4, 2, 7, 1, 5, 3, 6])

    # 3, 2, 1, 4, 5, 7, 6
    # return 3 (index of value 4 in a)
    assert 3 == kmax.partition(a, 0, len(a)-1)
    assert 0 == np.sum(a != np.array([3, 2, 1, 4, 5, 7, 6]))
    print('partition successful!')

if __name__ == "__main__":
    test_kth()
    test_get_knn_flag()
    test_partition()
