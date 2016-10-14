from samples.conf import LSFS
from samples.traintest import traintest
from samples.get_traintest import get_traintest
from samples.conf import get_filepath_in_folders
from samples.conf import fea_rank_read
from samples.conf import fea_rank_write
from samples.conf import *


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

        num_fea = 20
        fidx = range(num_fea)

        for gamma in gamma_list:
            fc = LSFS.LSFS(x_train[:,:num_fea].T, label_n1_to_nc(y_train), \
                           x_test[:,:num_fea].T, gama=gamma)
            rf = LSFS.feature_ranking(fc)

            exc_fun_label.append("{0}_gama({1})".format(LSFS.LSFS.__name__, gamma))
            feature_order_list.append(rf)

        fn = fn.format(test_foldth)
        new_path = output_path + path.split('data')[-1].strip()
        fea_rank_write(new_path, fn, feature_order_list, exc_fun_label, fidx)

    fr = fea_rank_read(['lsfs'])
    for feature_rank_table, path in fr:
        print(feature_rank_table.shape)

    print('{0} finish!'.format(__file__))


if __name__ == '__main__':
    LSFS_ranking()