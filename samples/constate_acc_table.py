from samples.conf import pd
from samples.conf import get_filepath_in_folders
from samples.conf import *

def constate_acc_table(path_list, fn):
    for folder_path in path_list:
        acc_path_list = get_filepath_in_folders(folder_path)

        print(folder_path)

        acc_table = pd.DataFrame()
        for path in acc_path_list:
            if 'baseline' in path or fn in path:
                continue

            table = pd.read_csv(path, index_col='index name')
            acc_table = pd.concat([acc_table, table])
            acc_table = pd.DataFrame(acc_table)

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

        acc_table.to_csv(folder_path + '/' + fn, header = True, index = True)


if __name__ == '__main__':
    path_list = [
        '../samples/result/Glass Identification(214x9)',
        '../samples/result/SPECTF Heart(188x44)',
        '../samples/result/LibrasMovement(360x90)',
        '../samples/result/Hill_Valley_without_noise_Testing(606x100)',
        '../samples/result/Hill_Valley_with_noise_Testing(606x100)',
        '../samples/result/Musk(476x166)',
        '../samples/result/LSVT_feature_names(126x310)'
    ]

    fn = 'all_acc_table.csv'

    constate_acc_table(path_list, fn)