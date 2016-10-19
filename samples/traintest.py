from  samples.conf import split_by_StratifiedKFold
from samples.conf import read_data
from samples.conf import create_path
from samples.conf import *


def traintest(op = "r", get_paths=False, n_folds = 10):
    """
    从指定的路径读取data.txt和row.label，
    然后划分成k折，输出到'./samles_data'文件夹中

    返回训练和测试数据，以及数据的路径
    """
    paths = None
    data_file_paths = 'data_selected_file_path.txt'
    fdname = 'data.txt'
    lname = 'row.label'

    output_path = './samles_data'

    # read data file paths
    with open(data_file_paths, "r") as result_file:
        paths = [p.strip('/|\n| ') for p in result_file.readlines() if len(p.strip('/|\n| ')) > 0]


    # crete output file name
    new_output_path = [output_path + p.split('data')[-1]  for p in paths]

    print(new_output_path)

    flag_list = []
    if op == 'w':
        rd = read_data(paths, fdname, lname)
        fileidx = -1
        for x, y in rd:
            fileidx += 1
            # test from 0.1, 0.2 to 0.9 of the origin data.
            # train from 0.9, 0.8 to 0.1 of the origin data.
            # train_bool = ~test_bool
            fn = new_output_path[fileidx] + '/'+ 'test_' + str(n_folds) + '.flag'

            create_path(new_output_path[fileidx])
            test_flag = split_by_StratifiedKFold(y, n_folds=n_folds)
            with open(fn, 'w+') as f:
                print('\n'.join([str(w) for w in test_flag]), file=f)
    elif op == 'r':
        for fileidx in range(len(new_output_path)):
            if len(new_output_path[fileidx]) == 0:
                continue
            fn = new_output_path[fileidx] + '/'+ 'test_' + str(n_folds) + '.flag'
            print("read: ", fn)
            flag_list.append(read_label(fn))

    if get_paths == True:
        return flag_list, paths
    else:
        return flag_list

