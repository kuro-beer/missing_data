# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import statsmodels.api as sm

from matplotlib import pyplot as plt
from scipy.stats import norm






def print_args(A, **kwargs):
    print(A)
    print(kwargs)
    print([key for key in kwargs.itervalues()])


print_args("XXX", **{"test": 1, "stat": 2})






