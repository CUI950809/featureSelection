from samples.conf import selected_data_by_flag
from samples.conf import read_data
from samples.conf import *

from samples.traintest import traintest


def test_traintest():
    """
    从data_selected_file_path.txt文件中指定的路径读取data.txt和row.label，
    然后划分成k折，输出到'./samles_data'文件夹中
    """
    n_folds = 10
    traintest(op="w", n_folds = n_folds)
    tt = traintest(op="r")
    for test_flag in tt:
        print(test_flag.shape, '\n', np.unique(test_flag))

if __name__ == '__main__':
    test_traintest()