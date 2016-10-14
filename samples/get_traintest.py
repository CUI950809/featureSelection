from samples.conf import *
from samples.traintest import traintest


def get_traintest(test_foldth = 3):
    data_file_paths = 'data_selected_file_path.txt'
    fdname = 'data.txt'
    lname = 'row.label'

    tt, paths = traintest(op="r", get_paths=True)

    fileidx = -1

    # read data file paths
    with open(data_file_paths, "r") as result_file:
        paths = [p.strip('/|\n| ') for p in result_file.readlines() if len(p.strip('/|\n| ')) > 0]

    rd = read_data(paths, fdname, lname)

    for x, y in rd:
        fileidx += 1
        train_flag = tt[fileidx] > test_foldth
        test_flag = tt[fileidx] <= test_foldth
        x_train, y_train, x_test, y_test = (
            x[train_flag],
            y[train_flag],
            x[test_flag],
            y[test_flag]
        )

        yield x_train, y_train, x_test, y_test, paths[fileidx]


def get_data():
    data_file_paths = 'data_selected_file_path.txt'
    fdname = 'data.txt'
    lname = 'row.label'

    tt, paths = traintest(op="r", get_paths=True)
    # new_paths = [ './ranking_result/' + p.split('data')[-1].strip() for p in paths]

    fileidx = -1

    # read data file paths
    with open(data_file_paths, "r") as result_file:
        paths = [p.strip('/|\n| ') for p in result_file.readlines() if len(p.strip('/|\n| ')) > 0]
    rd = read_data(paths, fdname, lname)

    for x, y in rd:
        fileidx += 1
        yield x, y, paths[fileidx]
