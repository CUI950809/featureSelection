from samples.conf import lsdf
from samples.traintest import traintest
from samples.get_traintest import get_traintest
from samples.conf import *

from utility.wrapper import LSDFTime

LSDFTime = LSDFTime

def save_lsdf_time(fn):
    fun_name = 'LSDF'
    mean_t = np.mean(LSDFTime)

    save_value = {'mean_t': mean_t}

    save_time(fn, fun_name, save_value)


def main():
    # new_paths = ['./ranking_result/' + p.split('data')[-1].strip() for p in paths]
    output_path = './ranking_result/'
    test_foldth = 3
    fn = 'lsdf_feature_rank_{0}.txt'.format(test_foldth)
    tt = get_traintest(test_foldth=test_foldth)
    for x_train, y_train, x_test, y_test, path in tt:
        feature_order_list = []
        exc_fun_label = []

        num_fea = x_train.shape[1]
        fidx = range(num_fea)

        fc = lsdf.lsdf(x_train[:,:num_fea], y_train, x_test[:,:num_fea])
        rf = lsdf.feature_ranking(fc)
        exc_fun_label.append("{0}".format(lsdf.lsdf.__name__))
        feature_order_list.append(rf)

        fn = fn.format(test_foldth)
        new_path = output_path + path.split('data')[-1].strip()
        fea_rank_write(new_path, fn, feature_order_list, exc_fun_label, fidx)

        # -------------------------save time--------------------------#
        fn = path.strip('/| |\n').split('/')[-1]
        save_lsdf_time(fn)

    fr = fea_rank_read(['lsdf'])
    for feature_rank_table, path in fr:
        print(feature_rank_table.shape)

    print('{0} finish!'.format(__file__))


if __name__ == '__main__':
    main()