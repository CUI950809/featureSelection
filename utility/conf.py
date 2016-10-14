#!usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import scipy as sp
import math
import time
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score