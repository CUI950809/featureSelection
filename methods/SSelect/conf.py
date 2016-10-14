import pandas as pd
import numpy as np
import math

from sklearn.metrics.pairwise import pairwise_distances

from utility.kmax import get_knn_flag
from utility.normalized_mutual_info_score import my_normalized_mutual_info_score as nmi