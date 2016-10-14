from samples.conf import svm
from samples.conf import accuracy_score
from samples.conf import gen_data
from samples.conf import laplacian_score
from samples.conf import *


def test_lsdf():
    tt = get_traintest()
    x_train, y_train, x_test, y_test, path = next(tt)
    lap_score = laplacian_score.laplacian_score(x_train, k = 10, t = 100)
    lap_score_rank = lsdf.feature_ranking(lap_score)

    num_fea = 100  # number of selected features
    idx = lap_score_rank[:num_fea]

    run_num = 10
    clf = svm.LinearSVC()

    accuracy = 0
    for i in range(run_num):
        clf.fit(x_train[:, idx], y_train)
        y_predict = clf.predict(x_test[:, idx])
        accuracy += accuracy_score(y_test, y_predict)
    print('Accuracy : {0}'.format(accuracy / run_num))


if __name__ == "__main__":
    test_lsdf()
