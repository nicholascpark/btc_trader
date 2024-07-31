import numpy as np


def safe_divide(a, b, fill_value=0):
    return np.divide(a, b, out=np.full_like(a, fill_value), where=b!=0)

def safe_log(x):
    return np.log(np.where(x > 0, x, np.nan))