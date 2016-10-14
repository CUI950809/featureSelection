from utility.conf import np


def compute_euclidean_distance(samples):
    """
    compute euclidean_distance between each sample in samples.

    Input
    -----
    samples: {numpy array}, shape {n_samples, n_features}

    Output
    ------
    dist: {numpy array}, shape {n_samples, n_samples}
    """

    # 计算S矩阵
    n_sampels = samples.shape[0]
    dist = np.zeros((n_sampels, n_sampels))
    for i in range(n_sampels):
        for j in range(n_sampels):
            value = np.linalg.norm(samples[i] - samples[j], ord='2')
            dist[i, j] = value
    return dist
