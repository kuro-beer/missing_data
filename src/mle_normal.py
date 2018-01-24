# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import statsmodels.api as sm

from matplotlib import pyplot as plt
from scipy.stats import norm, poisson, uniform
from scipy.optimize import minimize

missing_ratio = 0.3
sample_size = 10000
seed = 123

mu1 = 10
sigma1 = 5

def generate_sample(sample_size, random_state):
    np.random.seed(random_state)
    df = pd.DataFrame({
        "y1": np.random.normal(2, 5, sample_size),
        "y2": np.random.randn(sample_size),
        "r": np.ones(sample_size, dtype="int")
    })
    df["r"].where(df.y2 >= norm.ppf(missing_ratio), other=0, inplace=True)
    df["y1_obs"] = df["y1"].where(df.r == 1, other=np.nan)
    return df

def loglike(pdf, data, **kwargs):
    return np.sum(np.log(pdf(data, **kwargs)))

def _loglike_norm(x, *args):
    return -np.sum(np.log(norm.pdf(args[0], x[0], x[1]) + 1e-8))

def mle_norm(data, x0=None):
    if x0 == None:
        x0 = np.random.randn(2)

    estimates_ = minimize(_loglike_norm,
                          x0,
                          args=(data),
                          method='Nelder-Mead',
                          options={'maxiter': 100,
                                   'disp': True})
    return estimates_

def main():
    df = generate_sample(sample_size, seed)
    print(df.head(10))

    # model = sm.GLM(df.y1_obs.loc[~np.isnan(df.y1_obs)],
    #                np.ones(len(df.y1_obs.loc[~np.isnan(df.y1_obs)])),
    #                family=sm.families.Gaussian())
    # result = model.fit()
    # print(result.summary())

    params = {"loc": 0,
              "scale": 1}
    print(loglike(data=df.y1, pdf=norm.pdf, **params))


    d = np.random.random_integers(0, 10, 3)
    params = {"loc": 0,
              "scale": 10}
    print(d)
    print(loglike(data=d, pdf=uniform.pdf, **params))

    print(mle_norm(data=df.y1))


if __name__ == "__main__":
    main()
