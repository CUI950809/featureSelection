from samples.conf import cal_acc_tabel
from samples.conf import fea_rank_read
from samples.conf import get_traintest
from samples.conf import create_path

def test_cal_accracy():
    output_path = './result/'
    idx_array = range(10, 16, 5)
    run_num = 50
    test_foldth = 3

    dname_list = ['adenocarcinoma', 'brain', 'breast.2.class', 'breast.3.class']

    frr = fea_rank_read(['lsfs', 'fisher', 'lsdf', 'sselect', 'prpc', 'laplacian'])
    rank_paths = [];
    feature_rank_table_list = [];
    for feature_rank_table, path in frr:
        rank_paths.append(path)
        feature_rank_table_list.append(feature_rank_table)

    tt = get_traintest(test_foldth=test_foldth)

    fn = 'lsdf_feature_rank_{0}.txt'.format(test_foldth)

    for x_train, y_train, x_test, y_test, path in tt:
        data_name = None

        for dn in dname_list:
            if dn in path:
                data_name = dn
                break

        for idx, rp in enumerate(rank_paths):
            if data_name in rp:
                tmp_list = []
                for p in rank_paths[idx].split('/'):
                    tmp_list.extend(p.split('\\'))

                fun_name = tmp_list[-1].split('_')[0]

                feature_order_table = feature_rank_table_list[idx]

                num_fea = feature_rank_table.shape[1]
                fn = fun_name + '_acc_feas{0}_folds{1}.txt'.format(num_fea, test_foldth)

                new_path = output_path + path.split('data')[-1].strip()
                create_path(new_path)

                acc_table = cal_acc_tabel(x_train, y_train, x_test, y_test, \
                              feature_order_table, idx_array, run_num=run_num)
                acc_table.to_csv(new_path + '/' + fn, header=True, index=True)

        print("*************************")


if __name__ == '__main__':
    test_cal_accracy()