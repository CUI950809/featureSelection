from samples.conf import pd
from samples.conf import get_filepath_in_folders
from samples.conf import *


def constate_acc_table(in_path_list, out_path_list, fn):
    for fpidx, folder_path in enumerate(in_path_list):
        acc_path_list = get_filepath_in_folders(folder_path)

        print(folder_path)

        max_acc_table = pd.DataFrame()
        acc_table = pd.DataFrame()
        for path in acc_path_list:
            if 'baseline' in path or fn in path:
                continue

            table = pd.read_csv(path, index_col='index name')
            acc_table = pd.concat([acc_table, table])
            acc_table = pd.DataFrame(acc_table)

            idx = table.apply(np.sum, axis=1).argmax()
            max_acc_table = pd.concat((max_acc_table, pd.DataFrame(table.ix[idx, :]).T));
            max_acc_table = pd.DataFrame(max_acc_table)

        for path in acc_path_list:
            if 'baseline' not in path or fn in path:
                continue

            baseline = pd.read_csv(path, header=None, index_col=None).values
            index_name = ['baseline']
            columns = acc_table.columns
            num_col = acc_table.shape[1]
            baseline_table = pd.DataFrame(baseline.repeat(num_col).reshape(1,num_col),
                                          columns=columns, index=index_name)
            baseline_table.index.names = acc_table.index.names

            acc_table = pd.concat((baseline_table, acc_table))
            max_acc_table = pd.concat((baseline_table, max_acc_table))

        create_path(out_path_list[fpidx])
        acc_table.to_csv(out_path_list[fpidx] + '/' + fn + '.csv', header = True, index = True)
        max_acc_table.to_csv(out_path_list[fpidx] + '/' + 'max_' + fn + '.csv', header=True, index=True)

        plot_acc_arr(max_acc_table, picture_path=out_path_list[fpidx] + '/' + 'max_' + fn + '.png')



if __name__ == '__main__':
    path_list = [
        # '../samples/result/Glass Identification(214x9)',
        # '../samples/result/SPECTF Heart(188x44)',
        # '../samples/result/LibrasMovement(360x90)',
        # '../samples/result/Hill_Valley_without_noise_Testing(606x100)',
        # '../samples/result/Hill_Valley_with_noise_Testing(606x100)',
        # '../samples/result/Musk(476x166)',
        # '../samples/result/LSVT_feature_names(126x310)',

        '../samples/result/optdigits(5620x63)/',
        '../samples/result/MADELON(2000x600)/',
        '../samples/result/isolet5(7797x617)/',
        '../samples/result/multiple_feature(2000x649)/',
        '../samples/result/CNAE-9(1080x856)/',
        '../samples/result/colon(62x2000)/',
        '../samples/result/srbct(63x2308)/',
        '../samples/result/leukemia(38x3051)/',
        '../samples/result/lymphoma(62x4026)/',
        '../samples/result/breast.2.class(77x4869)/',
        '../samples/result/breast.3.class(95x4869)/',
        '../samples/result/nci(61x5244)/',
        '../samples/result/brain(42x5597)/',
        '../samples/result/prostate(102x6033)/',
        '../samples/result/adenocarcinoma(76x9868)/'
    ]

    output_path = './acc_result/'
    new_folder_path = [output_path + '/' + p.split('result')[-1] for p in path_list]

    fn = 'all_acc_table'

    constate_acc_table(path_list, new_folder_path, fn)
