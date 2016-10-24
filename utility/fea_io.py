from utility.conf import np
from utility.conf import pd
from utility.path_search import create_path
from utility.path_search import get_filepath_in_folders
from utility.my_plot import plot_acc_arr


feature_ranking_file_path = 'feature_ranking_file_path.txt'
feature_weight_file_path = 'feature_weight_file_path.txt'


def fea_rank_write(path, fn, \
                feature_order_list, exc_fun_label, fidx):
    """
    write features ranking to specify file.(path + fn).
    also the path (path) will write to file in feature_ranking_file_path.

    Input
    ----
    path: {str} the path where to write feature ranking.
    fn: {str} file name.
    feature_order_list: {list}, len is m, each element is numpy array shape {n_features,}.
    exc_fun_label: {numpy array}, shape {m,}. the index for feature ranking
    fidx: {numpy array}, shape {n_features,}

    Output
    ------
    None
    """
    num_fea = 0

    if len(feature_order_list) > 0:
        num_fea = len(feature_order_list[0])

    new_path = path + "/n{0}/".format(num_fea)

    create_path(new_path)

    with open(feature_ranking_file_path, "a+") as f:
        if new_path not in f.readlines():
            print(new_path, file=f)

    feature_order_table_path = new_path + fn
    feature_order_table = pd.DataFrame(data=np.array(feature_order_list), index=exc_fun_label, columns=fidx)
    feature_order_table.index.name='index name'
    print('write : ', feature_order_table_path)
    if feature_order_table_path != None:
        feature_order_table.to_csv(feature_order_table_path, header=True, index=True)


def fea_rank_read(select = None):
    """
    read features ranking from specify file, list in the file in path
    feature_ranking_file_path

    Input
    ----
    select: {list}. key str in list. it is use to select file to read.
    for example, if select = ['lsfs'], file in folder which list in feature_ranking_file_path.
    if feature_ranking_file_path consist './xxx/', this folder have lsfs_n50.txt and lsdf.txt.
    lsfs_n50 will be read.

    Output
    ------
    yield feature_order_table, file_path
    feature_order_table: {pandas DataFrame}, shape {m, n_features}
    file_path: {str}. featurr_order_table read from one file in file_path.
    """

    if select is None:
        select = ['all']

    select = [name.lower() for name in select]

    # read data file paths
    with open(feature_ranking_file_path, "r") as result_file:
        paths = [p.strip('/|\n| ') for p in result_file.readlines() if len(p.strip('/|\n| ')) > 0]

    # crete output file name
    new_paths = [p.strip('/|\n| ').split('data')[-1] for p in paths]

    have_read = []
    for path in new_paths:

        if path in have_read:
            continue

        have_read.append(path)

        file_paths = get_filepath_in_folders(path)
        for file_path in file_paths:

            flag = False
            for filename in select:
                if filename.lower() in file_path.lower():
                    flag = True

            if 'all' in select:
                flag = True

            if flag == False:
                continue

            print('read : ', file_path)
            feature_order_table = pd.read_csv(file_path, index_col='index name')
            # print(feature_order_table.values)
            yield feature_order_table, file_path


"""
写到这了
"""
def fea_weight_write(path, fn, \
                  feature_weight_list, exc_fun_label, fidx, \
                     sort_flag = False, reverse_flag = False):
    """
    write features ranking to specify file.(path + fn).
    also the path (path) will write to file in feature_weight_file_path.

    Input
    ----
    path: {str} the path where to write feature ranking.
    fn: {str} file name.
    feature_weight_list: {list}, len is m, each element is numpy array shape {n_features,}.
    exc_fun_label: {numpy array}, shape {m,}. the index for feature ranking
    fidx: {numpy array}, shape {n_features,}

    Output
    ------
    None
    """
    num_fea = 0

    if len(feature_weight_list) > 0:
        num_fea = len(feature_weight_list[0])

    # new_path = path + "/n{0}/".format(num_fea)
    new_path = path + "/"

    create_path(new_path)

    with open(feature_weight_file_path, "a+") as f:
        if new_path.strip('/') not in f.readlines():
            print(new_path, file=f)

    feature_weight_table_path = new_path + fn + '.csv'
    feature_weight_table = pd.DataFrame(data=np.array(feature_weight_list), index=exc_fun_label, columns=fidx)
    feature_weight_table.index.name = 'index name'
    print('write : ', feature_weight_table_path)
    if feature_weight_table_path != None:
        feature_weight_table.to_csv(feature_weight_table_path, header=True, index=True)

    plot_table = feature_weight_table
    if sort_flag == True:
        arr = plot_table.values
        new_arr = np.sort(arr, axis=1)
        if reverse_flag == True:
            for i in range(len(arr)):
                new_arr[i, :] = new_arr[i,::-1]
        plot_table = pd.DataFrame(np.array(new_arr),
                             index=feature_weight_table.index, columns=feature_weight_table.columns)
        plot_table.index.name = feature_weight_table.index.name

    plot_acc_arr(plot_table, picture_path=new_path + '/' + fn + '.png')



