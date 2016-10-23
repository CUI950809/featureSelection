#!usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances
from utility.kmax import get_knn_flag
from utility.wrapper import timeit
from utility.wrapper import reset_lsdf_global_value
from utility.my_exam import save_time