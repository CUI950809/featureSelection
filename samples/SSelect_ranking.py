from samples.conf import SSelect
from samples.conf import *
from samples.traintest import traintest
from samples.get_traintest import get_traintest


from utility.wrapper import SSelectTime

SSelectTime = SSelectTime

def save_SSelect_time(fn):
    fun_name = 'SSelect'
    mean_t = np.mean(SSelectTime)

    save_value = {'mean_t': mean_t}

    save_time(fn, fun_name, save_value)


def main():

    # new_paths = ['./ranking_result/' + p.split('data')[-1].strip() for p in paths]
    output_path = './ranking_result/'
    test_foldth = 3
    fn = 'SSelect_feature_rank_{0}.txt'.format(test_foldth)

    theta_list = [10**i for i in range(-3,4)]
    namuda_list = [i*0.1 for i in range(0,11)]

    tt = get_traintest(test_foldth=test_foldth)
    for x_train, y_train, x_test, y_test, path in tt:
        feature_order_list = []
        exc_fun_label = []

        num_fea = x_train.shape[1]
        fidx = range(num_fea)
        for theta in theta_list:
            for namuda in namuda_list:
                fc = SSelect.SSelect(x_train[:,:num_fea], y_train, x_test[:,:num_fea], theta = theta, namuda=namuda)
                rf = SSelect.feature_ranking(fc)
                exc_fun_label.append("{0}_theta({1})_namuda({2})".format(SSelect.SSelect.__name__, theta, namuda))
                feature_order_list.append(rf)

                # -------------------------save time--------------------------#
                fn = path.strip('/| |\n').split('/')[-1]
                save_SSelect_time(fn)

        fn = fn.format(test_foldth)
        new_path = output_path + path.split('data')[-1].strip()
        fea_rank_write(new_path, fn, feature_order_list, exc_fun_label, fidx)


    fr = fea_rank_read(['sselect'])
    for feature_rank_table, path in fr:
        print(feature_rank_table.shape)

    print('{0} finish!'.format(__file__))

if __name__ == '__main__':
    main()