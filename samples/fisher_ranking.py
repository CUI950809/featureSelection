from samples.conf import fisher_score
from samples.traintest import traintest
from samples.get_traintest import get_traintest
from samples.conf import *


def main():
    # new_paths = ['./ranking_result/' + p.split('data')[-1].strip() for p in paths]
    output_path = './ranking_result/'
    test_foldth = 3
    fn = 'fisher_feature_rank_{0}.txt'.format(test_foldth)
    tt = get_traintest(test_foldth = test_foldth)
    for x_train, y_train, x_test, y_test, path in tt:
        feature_order_list = []
        exc_fun_label = []

        num_fea = 20
        fidx = range(num_fea)

        fc = fisher_score.fisher_score(x_train[:,:num_fea], y_train)
        rf = fisher_score.feature_ranking(fc)

        exc_fun_label.append("{0}".format(fisher_score.fisher_score.__name__))
        feature_order_list.append(rf)

        fn = fn.format(test_foldth)
        new_path = output_path + path.split('data')[-1].strip()
        fea_rank_write(new_path, fn, feature_order_list, exc_fun_label, fidx)


    fr = fea_rank_read(['fisher'])
    for feature_rank_table, path in fr:
        print(feature_rank_table.shape)


    print('{0} finish!'.format(__file__))


if __name__ == '__main__':
    main()