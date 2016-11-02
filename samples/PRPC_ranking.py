from samples.conf import PRPC
from samples.conf import *
from samples.traintest import traintest
from samples.get_traintest import get_traintest


from utility.wrapper import PRPCTime

PRPCTime = PRPCTime

def save_PRPC_time(fn):
    fun_name = 'PRPC'
    mean_t = np.mean(PRPCTime)

    save_value = {'mean_t': mean_t}

    save_time(fn, fun_name, save_value)


def main():

    # new_paths = ['./ranking_result/' + p.split('data')[-1].strip() for p in paths]
    output_path = './ranking_result/'
    test_foldth = 3
    select_fea = 1000

    fn = 'PRPC_feature_rank_fold{0}_fea{1}.txt'

    tt = get_traintest(test_foldth=test_foldth)
    for x_train, y_train, x_test, y_test, path in tt:

        feature_order_list = []
        exc_fun_label = []

        num_fea = x_train.shape[1]

        actual_select_fea = select_fea
        if num_fea < select_fea:
            actual_select_fea = num_fea
        else:
            actual_select_fea = select_fea

        fn.format(test_foldth, actual_select_fea)

        fidx = range(actual_select_fea)

        rf = PRPC.PRPC(x_train[:,:num_fea], y_train, x_test[:,:num_fea], num=actual_select_fea)

        exc_fun_label.append("{0}".format(PRPC.PRPC.__name__))
        feature_order_list.append(rf)

        fn = fn.format(test_foldth, num_fea)
        new_path = output_path + path.split('data')[-1].strip()
        fea_rank_write(new_path, fn, feature_order_list, exc_fun_label, fidx)

        # -------------------------save time--------------------------#
        PRPC_time_fn = path.strip('/| |\n').split('/')[-1]
        save_PRPC_time(PRPC_time_fn)

    fr = fea_rank_read(['PRPC'])
    for feature_rank_table, path in fr:
        print(feature_rank_table.shape)

    print('{0} finish!'.format(__file__))

if __name__ == '__main__':
    main()