from samples.conf import svm
from samples.conf import pd
from samples.conf import accuracy_score
from samples.conf import get_traintest
from samples.conf import label_n1_to_nc
from samples.conf import plt
from samples.conf import np
from samples.conf import plot_acc_arr

from samples.conf import *

from samples.conf import LSFS
from utility.wrapper import LSFSObejectV, LSFSTime, LSFSGetWTime, LSFSCWTime, LSFSresult, LSFSWObejectV


LSFS.LSFS_NITER = 10
LSFS.W_NITER = 500
GetWTime = LSFSGetWTime
CWTime = LSFSCWTime
WObejectV = LSFSWObejectV
ObejectV = LSFSObejectV
t_list = LSFSTime
result = LSFSresult


"""
100:
CWTime :  0.00146126179039
GetWTime :  0.0588082319415
Accuracy : 0.7647058823529411

200:
CWTime :  0.00693430022355
GetWTime :  0.233652575151
Accuracy : 0.7647058823529411

300:
CWTime :  0.0145702458104
GetWTime :  0.478903751978
Accuracy : 0.7941176470588235

500:
CWTime :  0.057051783505
GetWTime :  1.82739611649
Accuracy : 0.7941176470588235

1000：
CWTime :  0.349255423923
GetWTime :  10.9484101281
Accuracy : 0.7941176470588235
"""



def compute_variation(arr, fun):
    """
    计算函数值的变化
    Input
    -----
    arr: {numpy array}, shape {n,}
    fun: 计算函数值的变化

    Output
    ------
    arr: {numpy array}, shape {n,}
    """
    new_arr = np.zeros(len(arr) - 1)
    for i in range(len(new_arr)):
        new_arr[i] = fun(arr[i + 1], arr[i])
    return new_arr


def time_isExit(x, column_dict):
    cd = column_dict
    cn = list(cd.keys())
    flag = True

    for key in cd:
        if key not in x.index or x[key] != cd[key]:
            flag = False
            break

    return flag


def save_time(fn, fun_name, save_value):
    time_file_name = 'exec_time.csv'
    columns = ['data name', 'fun name', 'which', 'time']

    time_table = pd.DataFrame(columns=columns)
    time_table.index.name = 'index name'
    if path_isExists(time_file_name):
        time_table = pd.read_csv(time_file_name)

    for which in save_value:
        t_table = pd.DataFrame(np.array([fn, fun_name, which, save_value[which]]).reshape(1, -1)
                               , columns=columns)

        value_dict = {columns[0]: fn, columns[1]: fun_name, columns[2]: which}
        flag = time_table.apply(lambda x: time_isExit(x, value_dict), axis=1)

        if flag.shape[0] == 0 or (flag.shape[0] != 0 and not flag.any()):
            time_table = time_table.append(t_table, ignore_index=True)

    time_table.to_csv(time_file_name, index=False)


def test_LSFS():
    tt = get_traintest()
    next(tt)
    x_train, y_train, x_test, y_test, path = next(tt)
    LSFS_score = LSFS.LSFS(x_train.T, label_n1_to_nc(y_train), x_test.T, gama=10**-1)
    LSFS_score_rank = LSFS.feature_ranking(LSFS_score)

    print('CWTime : ', np.mean(CWTime))
    print('GetWTime : ', np.mean(GetWTime))

    print('cnt CWTime : ', len(CWTime))
    print('cnt GetWTime : ', len(GetWTime))
    print('avg cnt CWTime : ', len(CWTime) / len(GetWTime))

    print('WObejectV : ', WObejectV)
    print('LSFSObejectV : ', LSFSObejectV)

    num_fea = 100  # number of selected features
    if num_fea > len(LSFS_score_rank):
        num_fea = int(len(LSFS_score_rank) / 2)
    idx = LSFS_score_rank[:num_fea]

    run_num = 10
    clf = svm.LinearSVC()

    accuracy = 0
    for i in range(run_num):
        clf.fit(x_train[:, idx], y_train)
        y_predict = clf.predict(x_test[:, idx])
        accuracy += accuracy_score(y_test, y_predict)
    print('Accuracy : {0}'.format(accuracy / run_num))


    #-------------------------save obj change--------------------------#
    objv_path = './obj_result/'
    output_path = objv_path + path.split('data')[-1].strip()

    wobjv_change = compute_variation(WObejectV, LSFS.value_variation)
    objv_chamge = compute_variation(ObejectV, LSFS.value_variation)

    wobjv_change = pd.DataFrame(wobjv_change.reshape(1, -1),
        index=['LSFSWObejectV'])

    objv_chamge = pd.DataFrame(objv_chamge.reshape(1, -1),
        index=['LSFSObejectV'])
    wobjv_change.index.name = 'index name'
    objv_chamge.index.name = 'index name'

    create_path(output_path)
    wobjv_change.to_csv(output_path + '/' + 'LSFSWObejectV.csv', header=True, index=True)
    objv_chamge.to_csv(output_path + '/' + 'LSFSObejectV.csv', header=True, index=True)
    plot_acc_arr(wobjv_change, xlabel="iter", ylabel='value',
                 picture_path=output_path + '/' + 'LSFSWObejectV.png')
    plot_acc_arr(objv_chamge, xlabel="iter", ylabel='value',
                 picture_path=output_path + '/' + 'LSFSObejectV.png')

    # -------------------------save time--------------------------#
    fn = path.strip('/| |\n').split('/')[-1]
    fun_name = 'LSFS'

    mean_w_time = np.mean(CWTime)
    mean_getw_time = np.mean(GetWTime)
    mean_t = np.mean(LSFSTime)
    sum_w_time = np.sum(CWTime)
    sum_getw_time = np.sum(GetWTime)

    save_value = {"mean_getw_time":mean_getw_time,
                  'mean_w_time':mean_w_time,
                  'mean_t':mean_t,
                  'sum_w_time':sum_w_time,
                  'sum_getw_time':sum_getw_time}

    save_time(fn, fun_name, save_value)


if __name__ == "__main__":
    test_LSFS()
