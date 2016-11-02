from samples.conf import pd
from samples.conf import get_filepath_in_folders
from samples.conf import *


def constate_fw_table(in_path_list, out_path_list, fn):
    for fpidx, folder_path in enumerate(path_list):
        acc_path_list = get_filepath_in_folders(folder_path)

        print(folder_path)

        acc_table = pd.DataFrame()
        for path in acc_path_list:
            table = pd.read_csv(path, index_col='index name')
            acc_table = pd.concat([acc_table, table])
            acc_table = pd.DataFrame(acc_table)

        create_path(out_path_list[fpidx])
        acc_table.to_csv(out_path_list[fpidx] + '/' + fn + '.csv', header = True, index = True)
        plot_acc_arr(acc_table, picture_path=out_path_list[fpidx] + '/' + 'max_' + fn + '.png')


if __name__ == '__main__':
    path_list = [
        '../samples/fw_result/Glass Identification(214x9)',
        '../samples/fw_result/SPECTF Heart(188x44)',
        '../samples/fw_result/LibrasMovement(360x90)',
        '../samples/fw_result/Hill_Valley_without_noise_Testing(606x100)',
        '../samples/fw_result/Hill_Valley_with_noise_Testing(606x100)',
        '../samples/fw_result/Musk(476x166)',
        '../samples/fw_result/LSVT_feature_names(126x310)'
    ]

    output_path = './fw_all_result/'
    new_folder_path = [output_path + '/' + p.split('result')[-1] for p in path_list]

    fn = 'all_fw_table'

    constate_fw_table(path_list, new_folder_path, fn)
