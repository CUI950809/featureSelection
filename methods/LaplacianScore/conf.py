#!usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
import sys
import time
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_distances
from utility.kmax import get_knn_flag

from utility.wrapper import timeit
from utility.wrapper import reset_LaplacianScore_global_value

