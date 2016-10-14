from samples.conf import cross_validation
from samples.conf import svm
from samples.conf import accuracy_score
from samples.conf import get_data
from samples.conf import *


def main():
    # load data
    gd = get_data()
    X,y, path = next(gd)

    n_samples, n_features = X.shape    # number of samples2 and number of features

    # split data into several folds
    num_folders = 3
    ss = cross_validation.StratifiedKFold(y, n_folds=num_folders, shuffle=True)

    # perform evaluation on classification task
    num_fea = 100    # number of selected features
    clf = svm.LinearSVC()    # linear SVM

    correct = 0
    for train, test in ss:
        # obtain the score of each feature on the training set
        score = fisher_score.fisher_score(X[train], y[train])

        # rank features in descending order according to score
        idx = fisher_score.feature_ranking(score)

        # obtain the dataset on the selected features
        selected_features = X[:, idx[0:num_fea]]

        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features[train], y[train])

        # predict the class labels of samples2 data
        y_predict = clf.predict(selected_features[test])

        # obtain the classification accuracy on the samples2 data
        acc = accuracy_score(y[test], y_predict)
        correct += acc

    # output the average classification accuracy over all 10 folds
    print('Accuracy:', float(correct)/num_folders)


if __name__ == '__main__':
    main()
