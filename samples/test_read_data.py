from samples.conf import read_data
from samples.conf import *


def test_read_data():
    path = './test_data'
    tmp_file = path + '/' + 'tmp13123124.txt'
    paths = None

    fdname='data.txt'
    lname='row.label'

    create_path(path)

    d = np.array([[1,2], [3, 4]])
    l = np.array([1,2])

    with open(tmp_file, 'w+') as f:
        f.write('./test_data/')

    with open(path + '/' + fdname, 'w+') as f:
        for row in d:
            print(' '.join([str(w) for w in row]), file=f)

    with open(path + '/' + lname, 'w+') as f:
        for v in l:
            print(v, file=f)

    with open(tmp_file, "r") as result_file:
        paths = result_file.readlines()

    print(paths)

    rd = read_data(paths, fdname, lname)
    x, y = next(rd)

    assert np.all(d == x) == True
    assert np.all(l == y) == True
    print("test_read_data successfully!")

if __name__ == "__main__":
    test_read_data()
