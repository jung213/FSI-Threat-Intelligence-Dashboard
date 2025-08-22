import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(expected: np.ndarray, actual: np.ndarray, bins=10):
    e_perc = pd.cut(expected, bins=bins, retbins=False, include_lowest=True)
    a_perc = pd.cut(actual,   bins=pd.IntervalIndex(e_perc.cat.categories), include_lowest=True)
    e_dist = e_perc.value_counts(normalize=True).sort_index()
    a_dist = a_perc.value_counts(normalize=True).sort_index()
    # 안정성 보정
    e_dist = e_dist.replace(0, 1e-6)
    a_dist = a_dist.replace(0, 1e-6)
    return np.sum((a_dist - e_dist) * np.log(a_dist / e_dist))

def ks_stat(expected: np.ndarray, actual: np.ndarray):
    return ks_2samp(expected, actual).statistic
