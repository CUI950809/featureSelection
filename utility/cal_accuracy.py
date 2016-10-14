from utility.conf import svm
from utility.conf import accuracy_score
from utility.conf import np
from utility.conf import pd


def run_acc(x_train, y_train, x_test, y_test, run_num = 10):
    """
    calculate accuracy.

    Input
    -----
    x_train: {numpy array}, shape {n_samples, n_features}
    y_train: {numpy array}, shape {n_samples,}
    x_test: {numpy array}, shape {n2_samples, n_features}
    y_test: {numpy array}, shape {n2_samples,}

    Output
    ------
    avg_accuracy: {float}
    """

    model = svm.LinearSVC(loss='hinge')
    accuracy = 0
    for i in range(run_num):
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        accuracy += accuracy_score(y_test, y_predict)

    avg_accuracy = accuracy / run_num
    return avg_accuracy


def cal_many_acc_by_idx(x_train, y_train, x_test, y_test, \
                         feature_order, idx_array, run_num = 10):
    """
    calculate accuracy according feature number in idx_array

    Input
    -----
    x_train: {numpy array}, shape {n_samples, n_features}
    y_train: {numpy array}, shape {n_samples,}
    x_test: {numpy array}, shape {n2_samples, n_features}
    y_test: {numpy array}, shape {n2_samples,}
    feature_order: {numpy array}, shape {n_features,}
    idx_array: {numpy array}, shape {n,}

    Output
    ------
    acc_array: {numpy array}, shape {n,}

    """
    idx_array = np.array(idx_array)
    acc_array = np.zeros(idx_array.shape)
    for i, num_fea in enumerate(idx_array):
        idx = feature_order[:num_fea]
        new_x_train, new_x_test = x_train[:, idx], x_test[:, idx]
        new_y_train, new_y_test = y_train, y_test
        a = run_acc(new_x_train, new_y_train, new_x_test, new_y_test, run_num=run_num)
        acc_array[i] = a
    return acc_array


def cal_acc_tabel(x_train, y_train, x_test, y_test, \
                  feature_order_table, idx_array, run_num=10):
    acc_array_list = []

    labels = feature_order_table.index.get_values()
    labels_name = feature_order_table.index.name

    for label in labels:
        feature_order = feature_order_table.ix[label, :]
        acc_array = cal_many_acc_by_idx(x_train, y_train, x_test, y_test,\
                                   feature_order.get_values(), idx_array = idx_array, run_num=run_num)
        acc_array_list.append(acc_array)

    acc_array_table = pd.DataFrame(data=acc_array_list, index=labels, columns=idx_array)
    acc_array_table.index.name = labels_name

    return acc_array_table






