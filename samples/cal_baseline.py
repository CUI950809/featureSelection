from samples.conf import cross_validation
from samples.conf import svm
from samples.conf import accuracy_score
from samples.conf import get_data
from samples.conf import *


def cal_baseline():
    num_folders = 3
    output_path = './result/'
    fn = 'baseline_accuracy_folders_{0}.txt'.format(num_folders)

    # load data
    gd = get_data()
    for X, y, path in gd:

        ss = cross_validation.StratifiedKFold(y, n_folds=num_folders, shuffle=True)

        # perform evaluation on classification task
        clf = svm.LinearSVC()    # linear SVM

        correct = 0
        for train, test in ss:
            # obtain the score of each feature on the training set
            score = fisher_score.fisher_score(X[train], y[train])

            # rank features in descending order according to score
            idx = fisher_score.feature_ranking(score)

            # train a classification model with the selected features on the training dataset
            clf.fit(X[train], y[train])

            # predict the class labels of samples2 data
            y_predict = clf.predict(X[test])

            # obtain the classification accuracy on the samples2 data
            acc = accuracy_score(y[test], y_predict)
            correct += acc

        # output the average classification accuracy over all k folds
        avg_accuracy = correct*1.0/num_folders

        new_path = output_path + path.split('data')[-1].strip()
        create_path(new_path)
        with open(new_path + '/' + fn, 'w+') as f:
            print(avg_accuracy, file=f)

    print('{0} finish!'.format(__file__))


if __name__ == "__main__":
    cal_baseline()