import shutil
import os

def moveFileto(sourceDir,  targetDir):
     shutil.copy(sourceDir,  targetDir)

def execfile(path):
    os.system('python {0}'.format(path))
    print(path, '     execute finished!')

target_file_path = './data_selected_file_path.txt'
file_path1_small = './data_selected_file_path_small.txt'
file_path_big = './data_selected_file_path_big.txt'
file_path_all = './data_selected_file_path_all.txt'


# 对数据进行划分
# moveFileto(file_path_all,  target_file_path)
# execfile('./test_traintest.py')


# 进行特征选择
execpath_list = {
    0:'./cal_baseline.py',
    1:'./fisher_ranking.py',
    2:'./laplacian_score_ranking.py',
    3:'./lsdf_ranking.py',
    4:'./LSFS_ranking.py',
    5:'./PRPC_ranking.py',
    6:'./SSelect_ranking.py'
}

evaluate_list = {100:'./test_cal_accuracy.py'}

# 小数据特征选择
moveFileto(file_path1_small,  target_file_path)
for key in execpath_list.keys():
    execfile(execpath_list[key])
    print('{0} finished!'.format(key))

for key in evaluate_list:
    execfile(evaluate_list[key])
    print('{0} finished!'.format(key))

# execfile('constate_acc_table.py')

