from samples.conf import LSFS
from samples.traintest import traintest
from samples.get_traintest import get_traintest
from samples.conf import get_filepath_in_folders
from samples.conf import fea_rank_read
from samples.conf import fea_rank_write
from samples.conf import *


from utility.wrapper import LSFSTime, LSFSFW

LSFSTime = LSFSTime
LSFSFW = LSFSFW


def save_LSFS_time(fn):
    fun_name = 'LSFS'
    mean_t = np.mean(LSFSTime)
    save_value = {'mean_t': mean_t}
    save_time(fn, fun_name, save_value)


def save_lsfs_fw(output_path):
    name = 'LSFSFW'
    if len(LSFSFW) > 1:
        print("too many result")
    elif len(LSFSFW) == 1:
        save_objectv(LSFSFW[0], name, output_path, sort_flag=True, reverse_flag=True)


def LSFS_ranking():
    # new_paths = ['./ranking_result/' + p.split('data')[-1].strip() for p in paths]
    output_path = './ranking_result/'
    fn = '/LSFS_feature_rank_{0}.txt'
    test_foldth = 3
    gamma_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    tt = get_traintest(test_foldth=test_foldth)

    for x_train, y_train, x_test, y_test, path in tt:
        feature_order_list = []
        exc_fun_label = []

        num_fea = x_train.shape[1]
        fidx = range(num_fea)

        for gamma in gamma_list:
            fc = LSFS.LSFS(x_train[:,:num_fea].T, label_n1_to_nc(y_train), \
                           x_test[:,:num_fea].T, gama=gamma)
            rf = LSFS.feature_ranking(fc)

            exc_fun_label.append("{0}_gama({1})".format(LSFS.LSFS.__name__, gamma))
            feature_order_list.append(rf)

            # -------------------------save time--------------------------#
            fn = path.strip('/| |\n').split('/')[-1]
            save_LSFS_time(fn)

            # -------------------------save feature weight result--------------------------#
            fw_path = './fw_result/LSFS/'
            output_path = fw_path + path.split('data')[-1].strip()
            save_lsfs_fw(output_path + '/' + str(gamma) + '/')

        fn = fn.format(test_foldth)
        new_path = output_path + path.split('data')[-1].strip()
        fea_rank_write(new_path, fn, feature_order_list, exc_fun_label, fidx)

    fr = fea_rank_read(['lsfs'])
    for feature_rank_table, path in fr:
        print(feature_rank_table.shape)

    print('{0} finish!'.format(__file__))


if __name__ == '__main__':
    LSFS_ranking()