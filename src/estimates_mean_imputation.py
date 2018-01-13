# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import t, sem
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = pd.read_pickle('../data/sample_data.pkl')

    x1_mean = df.x1.loc[df.x1_case1 == 1].mean()

    df["x1_imputed"] = df.x1*df.x1_case1 + x1_mean*(1-df.x1_case1)
    df.x1.hist(bins=25, color="lightgray")
    df.x1_imputed.hist(bins=25, color="salmon", alpha=0.5)
    plt.legend()
    plt.savefig("../fig/histogram_mean_imputation.png", format="png")



    arr = df.x1.loc[df.x1_case1 == 1].values
    m = np.mean(arr)
    e = np.var(arr, ddof=1)
    print("Mean: {0:.4f}".format(m))
    print("95%CI:", t.interval(alpha=0.95, df=len(arr), loc=m, scale=e))

    arr = df.x1_imputed.values
    m = np.mean(arr)
    e = np.var(arr, ddof=1)
    print("Mean: {0:.4f}".format(m))
    print("95%CI:", t.interval(alpha=0.95, df=len(arr), loc=m, scale=e))

    arr = df.x1.values
    m = np.mean(arr)
    e = np.var(arr, ddof=1)
    print("Mean: {0:.4f}".format(m))
    print("95%CI:", t.interval(alpha=0.95, df=len(arr), loc=m, scale=e))

