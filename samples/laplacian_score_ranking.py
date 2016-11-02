from samples.conf import laplacian_score
from samples.conf import *
from samples.traintest import traintest
from samples.get_traintest import get_traintest


from utility.wrapper import LaplacianScoreTime

LaplacianScoreTime = LaplacianScoreTime

def save_LaplacianScore_time(fn):
    fun_name = 'LaplacianScore'
    mean_t = np.mean(LaplacianScoreTime)

    save_value = {'mean_t': mean_t}

    save_time(fn, fun_name, save_value)



def main():
    # new_paths = ['./ranking_result/' + p.split('data')[-1].strip() for p in paths]
    output_path = './ranking_result/'
    test_foldth = 3
    fn = 'laplacian_score_feature_rank_{0}.txt'.format(test_foldth)

    k = 10
    # t_list = [10**i for i in range(-3,4)]
    t_list = [0.01, 0.1, 1]

    tt = get_traintest(test_foldth=test_foldth)
    for x_train, y_train, x_test, y_test, path in tt:
        feature_order_list = []
        exc_fun_label = []

        num_fea = x_train.shape[1]
        fidx = range(num_fea)


        for t in t_list:
            lap_score = laplacian_score.laplacian_score(x_train[:, :num_fea], k = k, t = t)
            lap_score_ranking = laplacian_score.feature_ranking(lap_score)
            exc_fun_label.append("{0}_t({1})".format(laplacian_score.laplacian_score.__name__, t))
            feature_order_list.append(lap_score_ranking)

            # -------------------------save time--------------------------#
            LaplacianScore_fn = path.strip('/| |\n').split('/')[-1]
            save_LaplacianScore_time(LaplacianScore_fn)

        fn = fn.format(test_foldth)
        new_path = output_path + path.split('data')[-1].strip()
        fea_rank_write(new_path, fn, feature_order_list, exc_fun_label, fidx)


    fr = fea_rank_read(['laplacian'])
    for feature_rank_table, path in fr:
        print(feature_rank_table.shape)

    print('{0} finish!'.format(__file__))

if __name__ == '__main__':
    main()