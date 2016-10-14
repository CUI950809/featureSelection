from samples.conf import selected_data_by_flag
from samples.conf import read_data
from samples.conf import *

from samples.traintest import traintest


def test_traintest():
    traintest(op="w")
    tt = traintest(op="r")
    for test_flag in tt:
        print(test_flag.shape, '\n', np.unique(test_flag))

if __name__ == '__main__':
    test_traintest()