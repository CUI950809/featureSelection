from utility.path_search import path_isExists
from utility.path_search import create_path
from utility.conf import np
from utility.conf import pd
from utility.my_plot import plot_acc_arr

def compute_variation(arr, fun):
    """
    计算函数值的变化
    Input
    -----
    arr: {numpy array}, shape {n,}
    fun: 计算函数值的变化

    Output
    ------
    arr: {numpy array}, shape {n,}
    """
    import copy
    if len(arr) <= 1:
        return copy.deepcopy(arr)

    new_arr = np.zeros(len(arr) - 1)
    for i in range(len(new_arr)):
        new_arr[i] = fun(arr[i + 1], arr[i])
    return new_arr

def time_isExit(x, column_dict):
    cd = column_dict
    cn = list(cd.keys())
    flag = True

    for key in cd:
        if key not in x.index or x[key] != cd[key]:
            flag = False
            break

    return flag

def save_time(fn, fun_name, save_value):
    time_file_name = 'exec_time.csv'
    columns = ['data name', 'fun name', 'which', 'time']

    time_table = pd.DataFrame(columns=columns)
    time_table.index.name = 'index name'
    if path_isExists(time_file_name):
        time_table = pd.read_csv(time_file_name)

    for which in save_value:
        t_table = pd.DataFrame(np.array([fn, fun_name, which, save_value[which]]).reshape(1, -1)
                               , columns=columns)

        value_dict = {columns[0]: fn, columns[1]: fun_name, columns[2]: which}
        flag = time_table.apply(lambda x: time_isExit(x, value_dict), axis=1)

        if flag.shape[0] == 0 or (flag.shape[0] != 0 and not flag.any()):
            time_table = time_table.append(t_table, ignore_index=True)
        elif flag.shape[0] != 0 and flag.any():
            # print("equal")
            if time_table.ix[flag, columns[3]] is np.NaN:
                time_table.ix[flag, columns[3]] = 0

            time_table.ix[flag, columns[3]] = (time_table.ix[flag, columns[3]] + save_value[which])/2.0
            # print(time_table.ix[flag, :])

    time_table.to_csv(time_file_name, index=False)

def save_objectv(arr, name, output_path, sort_flag = False, reverse_flag = False):
    create_path(output_path)
    table = pd.DataFrame(np.array(arr).reshape(1, -1),
        index=[name])
    table.index.name = 'index name'
    table.to_csv(output_path + '/' + name + '.csv', header=True, index=True)

    plot_table = table
    if sort_flag == True:
        new_arr = sorted(arr)
        if reverse_flag == True:
            new_arr = new_arr[::-1]
        plot_table = pd.DataFrame(np.array(new_arr).reshape(1, -1),
                             index=[name])
        plot_table.index.name = 'index name'

    plot_acc_arr(plot_table, xlabel="iter", ylabel='value',
                 picture_path=output_path + '/' + name + '.png')

