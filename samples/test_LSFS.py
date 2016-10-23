from samples.conf import svm
from samples.conf import pd
from samples.conf import accuracy_score
from samples.conf import get_traintest
from samples.conf import label_n1_to_nc
from samples.conf import plt
from samples.conf import np
from samples.conf import plot_acc_arr
from samples.conf import save_time
from samples.conf import save_objectv
from samples.conf import compute_variation

from samples.conf import *

from samples.conf import LSFS
from utility.wrapper import LSFSObejectV, LSFSTime, LSFSGetWTime
from utility.wrapper import LSFSCWTime, LSFSFW, LSFSWObejectV

LSFS.LSFS_NITER = 10
LSFS.W_NITER = 50
GetWTime = LSFSGetWTime
CWTime = LSFSCWTime
WObejectV = LSFSWObejectV
ObejectV = LSFSObejectV
LSFSTime = LSFSTime
LSFSFW = LSFSFW


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


def save_lsfs_objv(output_path):
    w_name = 'LSFSWObV'
    v_name = 'LSFSObV'
    wc_change_name = 'LSFSWObVChange'
    v_change_name = 'LSFSObVChange'

    objv_change = compute_variation(ObejectV, LSFS.value_variation)
    wobjv_change = compute_variation(WObejectV, LSFS.value_variation)

    save_objectv(objv_change, v_change_name, output_path)
    save_objectv(wobjv_change, wc_change_name, output_path)

    save_objectv(ObejectV, v_name, output_path)
    save_objectv(WObejectV, w_name, output_path)


def save_lsfs_time(fn):
    fun_name = 'LSFS'
    mean_w_time = np.mean(CWTime)
    mean_getw_time = np.mean(GetWTime)
    mean_t = np.mean(LSFSTime)
    sum_w_time = np.sum(CWTime)
    sum_getw_time = np.sum(GetWTime)

    save_value = {"mean_getw_time": mean_getw_time,
                  'mean_w_time': mean_w_time,
                  'mean_t': mean_t,
                  'sum_w_time': sum_w_time,
                  'sum_getw_time': sum_getw_time}

    save_time(fn, fun_name, save_value)


def save_lsfs_fw(output_path):
    name = 'LSFSFW'
    if len(LSFSFW) > 1:
        print("too many result")
    elif len(LSFSFW) == 1:
        save_objectv(LSFSFW[0], name, output_path, sort_flag=True, reverse_flag=True)


def test_result_time_and_obj(path, gama):
    # -------------------------save obj change--------------------------#
    objv_path = './test/obj_result/LSFS/'
    output_path = objv_path + path.split('data')[-1].strip()
    save_lsfs_objv(output_path + '/' + str(gama) + '/')

    # -------------------------save result--------------------------#
    fw_path = './test/fw_result/LSFS/'
    output_path = fw_path + path.split('data')[-1].strip()
    save_lsfs_fw(output_path + '/' + str(gama) + '/')

    # -------------------------save time--------------------------#
    fn = path.strip('/| |\n').split('/')[-1]
    save_lsfs_time(fn)


def test_LSFS():
    tt = get_traintest()
    gamma_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for x_train, y_train, x_test, y_test, path in tt:
        for gama in gamma_list:
            LSFS_score = LSFS.LSFS(x_train.T, label_n1_to_nc(y_train), x_test.T, gama=gama)
            test_result_time_and_obj(path, gama)


if __name__ == "__main__":
    test_LSFS()
