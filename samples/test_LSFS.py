from samples.conf import svm
from samples.conf import pd
from samples.conf import accuracy_score
from samples.conf import gen_data
from samples.conf import LSFS
from samples.conf import *


"""
100:
CWTime :  0.00146126179039
GetWTime :  0.0588082319415
Accuracy : 0.7647058823529411

200:
CWTime :  0.00693430022355
GetWTime :  0.233652575151
Accuracy : 0.7647058823529411

300:
CWTime :  0.0145702458104
GetWTime :  0.478903751978
Accuracy : 0.7941176470588235

500:
CWTime :  0.057051783505
GetWTime :  1.82739611649
Accuracy : 0.7941176470588235

1000：
CWTime :  0.349255423923
GetWTime :  10.9484101281
Accuracy : 0.7941176470588235
"""


def test_LSFS():
    tt = get_traintest()
    x_train, y_train, x_test, y_test, path = next(tt)
    LSFS_score = LSFS.LSFS(x_train[:, :100].T, label_n1_to_nc(y_train), x_test[:, :100].T, gama=10**-1)
    LSFS_score_rank = LSFS.feature_ranking(LSFS_score)

    print('CWTime : ', np.mean(LSFS.CWTime))
    print('GetWTime : ', np.mean(LSFS.GetWTime))

    print('cnt CWTime : ', len(LSFS.CWTime))
    print('cnt GetWTime : ', len(LSFS.GetWTime))
    print('avg cnt CWTime : ', len(LSFS.CWTime) / len(LSFS.GetWTime))

    print('WObejectV : ', LSFS.WObejectV)
    print('LSFSObejectV : ', LSFS.LSFSObejectV)

    num_fea = 100  # number of selected features
    idx = LSFS_score_rank[:num_fea]

    run_num = 10
    clf = svm.LinearSVC()

    accuracy = 0
    for i in range(run_num):
        clf.fit(x_train[:, idx], y_train)
        y_predict = clf.predict(x_test[:, idx])
        accuracy += accuracy_score(y_test, y_predict)
    print('Accuracy : {0}'.format(accuracy / run_num))

    #----------------------------#
    new_WObejectV = np.zeros(len(LSFS.WObejectV) - 1)
    new_LSFSObejectV = np.zeros(len(LSFS.LSFSObejectV) - 1)
    for i in range(len(new_WObejectV)):
        new_WObejectV[i] = LSFS.value_variation(LSFS.WObejectV[i + 1], LSFS.WObejectV[i])

    for i in range(len(new_LSFSObejectV)):
        new_LSFSObejectV[i] = LSFS.value_variation(LSFS.LSFSObejectV[i + 1], LSFS.LSFSObejectV[i])

    new_WObejectV = pd.DataFrame(new_WObejectV)
    new_LSFSObejectV = pd.DataFrame(new_LSFSObejectV)

    new_WObejectV.plot()
    new_LSFSObejectV.plot()
    plt.show()


if __name__ == "__main__":
    test_LSFS()
