from samples.conf import cal_acc_tabel
from samples.conf import fea_rank_read
from samples.conf import get_traintest
from samples.conf import create_path

def test_cal_accracy():
    output_path = './result/'
    run_num = 50
    test_foldth = 3

    # dname_list = ['adenocarcinoma', 'brain', 'breast.2.class', 'breast.3.class']
    dname_list = {
    'Glass Identification(214x9)':[1*i for i in range(1,10)],
    'SPECTF Heart(188x44)':[5*i for i in range(1,9)],
    'LibrasMovement(360x90)':[5*i for i in range(1,11)],
    'Hill_Valley_without_noise_Testing(606x100)':[5*i for i in range(1,11)],
    'Hill_Valley_with_noise_Testing(606x100)':[5*i for i in range(1,11)],
    'Musk(476x166)':[10*i for i in range(1,11)],
    'LSVT_feature_names(126x310)':[20*i for i in range(1,11)]
    }

    frr = fea_rank_read(['lsfs', 'fisher', 'lsdf', 'sselect', 'prpc', 'laplacian'])
    rank_paths = [];
    feature_rank_table_list = [];
    for feature_rank_table, path in frr:
        rank_paths.append(path)
        feature_rank_table_list.append(feature_rank_table)

    tt = get_traintest(test_foldth=test_foldth)

    for x_train, y_train, x_test, y_test, path in tt:
        data_name = None

        for dn in dname_list:
            if dn in path:
                data_name = dn
                break

        for idx, rp in enumerate(rank_paths):
            if data_name is not None and data_name in rp:
                idx_array = dname_list[data_name]
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