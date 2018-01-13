# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

missing_ratio = 0.3
sample_size = 1000
seed = 121

if __name__ == "__main__":
    np.random.seed(seed)
    x1 = np.random.randn(sample_size)
    x2 = np.random.randn(sample_size)
    y_COMP = 5 + 2 * x1 - 3 * x2 + np.random.randn(sample_size)
    y_MCAR =
    print(x1[:10])
    print(x2[:10])
    print(y_COMP[:10])



