#!usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
import scipy.io
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

from methods.FisherScore import fisher_score
from methods.LSDF import lsdf
from methods.LaplacianScore import laplacian_score
from methods.SSelect import SSelect
from methods.PRPC import PRPC
from methods.LSFS import LSFS

from utility import dist_function as dfun
from utility import kmax
from utility.gen_data import gen_data
from utility.read_data import read_data
from utility.read_data import read_label
from utility.read_data import selected_data_by_flag
from utility.split_data import split_by_StratifiedKFold
from utility.read_data import label_n1_to_nc
from utility.path_search import path_isExists
from utility.path_search import create_path
from utility.path_search import get_filepath_in_folders

from utility.fea_io import fea_rank_read
from utility.fea_io import fea_rank_write
from utility.fea_io import fea_weight_write

from utility.cal_accuracy import cal_acc_tabel
from utility.my_plot import plot_acc_arr
from utility.my_exam import save_time
from utility.my_exam import save_objectv
from utility.my_exam import compute_variation

from samples.traintest import traintest
from samples.get_traintest import get_traintest
from samples.get_traintest import get_data