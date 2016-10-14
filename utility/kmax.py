from utility.conf import np
from utility.conf import warnings


def partition(arr, low, high):
    """
    parttion the values between arr[low] and arr[high](Containing equal low and high) to two
    partition, the values will be left par, if it is smaller
    than arr[low]. the values will be left par, if it is
    bigger than arr[low].

    notes:
    arr outer the partition function will be change.

    Input
    -----
    arr: {numpy array}, shape {n,}
    low: index of start value.
    high: index of end value.

    Output
    ------
    low: {int}. the index of origin arr[low] in eventually arr.
    """
    key = arr[low]

    while low < high:
        while low < high and arr[high] >= key:
            high -= 1
        arr[low] = arr[high]

        while low < high and arr[low] <= key:
            low += 1
        arr[high] = arr[low]

    arr[low] = key
    return low


def quicksort(arr, low, high):
    """
    sort the values between arr[low] and arr[high].(Containing equal low and high)

    notes:
    arr outer the partition function will be change.

    Input
    -----
    arr: {numpy array}, shape {n,}
    low: index of start value.
    high: index of end value.

    Output
    ------
    """

    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)


def get_knn(data, k=1):

    """
    get knn from each row in data matric

    Input
    -----
    data: {numpy array}, shape {m, n}
    k: k nearest neighbors, default k = 1

    Output
    -----
    knn_data: {numpy array}, shape {m, n}
    """
    knn_data = np.zeros(data.shape)
    knn_flag = get_knn_flag(data, k)
    knn_data[knn_flag] = data[knn_flag]
    return knn_data


def get_knn_flag(data, k=1):
    """
    get knn bool flag from each row in data matric

    Input
    -----
    data: {numpy array}, shape {m, n}
    k: k nearest neighbors, default k = 1

    Output
    -----
    knn_flag: {numpy bool array}, shape {m, n}
    """

    n, m = data.shape
    knn_flag = np.ones((n, m))

    if m < k:
        k = int(m * 0.5)
        warnings.warn("k大于样本数，k被设置成样本数的一半")

    for i, row in enumerate(data):
        row2 = row.copy()
        k_value = k_th(row2, 0, len(row2)-1, k)

        need_cnt = k - (len(row2) - np.sum(row2 >= k_value))

        if need_cnt < 0:
            raise ValueError("error: need_cnt should be positive integer,"
                             " but need_cnt get value {1}"
                             "".format(need_cnt))

        knn_flag[i:i+1, row2 >= k_value] = 0

        # index return is (ndarray, dtype), get index value through index[0]
        index = np.where(row2 == k_value)

        if len(index[0]) < need_cnt:
            raise ValueError("error: len(index): {0} >= need_cnt: {1}"
                             "".format(len(index), need_cnt))

        np.random.shuffle(index[0])

        knn_flag[i:i+1, index[0][:need_cnt]] = 1

    return knn_flag.astype(np.bool)


def k_th(arr, low, high, k, it_cnt=None):
    """
    search the k smallest value between arr[low] and arr[high].(Containing equal low and high)

    Input
    -----
    arr: {numpy array}, shape {n,}
    low: index of start value.
    high: index of end value.
    k: k smallest value
    it_cnt: record the round of iterator

    Output
    ------


    """
    import copy

    # 这步不能少，否则会改变外部数据
    if it_cnt is None:
        it_cnt = 0
        arr = copy.deepcopy(arr)

    it_cnt += 1

    if high - low + 1 < k or k < 1:
        raise ValueError("error : high - low + 1 < k or k < 1 \n"
                         "get  {0} - {1} + 1 < {2} or {2} < 1"
                         "".format(high, low, k))

    pos = partition(arr, low, high)
    cnt = pos - low + 1

    if cnt == k:
        return arr[pos]
    elif cnt < k:
        return k_th(arr, pos + 1, high, k - cnt, it_cnt)
    else:
        return k_th(arr, low, pos, k, it_cnt)
