from samples.conf import svm
from samples.conf import accuracy_score
from samples.conf import gen_data
from samples.conf import SSelect
from samples.conf import *


def test_SSelect():
    tt = get_traintest()
    x_train, y_train, x_test, y_test, path = next(tt)
    x_train, y_train, x_test, y_test, path = next(tt)
    SSelect_score = SSelect.SSelect(x_train, y_train, x_test)
    SSelect_score_rank = SSelect.feature_ranking(SSelect_score)

    num_fea = 100  # number of selected features
    selected_fea = SSelect_score_rank[:num_fea]

    clf = svm.LinearSVC()
    clf.fit(x_train[:, selected_fea], y_train)
    y_predict = clf.predict(x_test[:, selected_fea])
    accuracy = accuracy_score(y_test, y_predict)
    print('Accuracy : {0}'.format(accuracy))


if __name__ == "__main__":
    test_SSelect()
