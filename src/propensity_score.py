# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols, glm

sample_size = 1000
seed = 123

def gen_sample(sample_size, seed):
    np.random.seed(seed)
    z1, z2, z3 = np.random.randn(3, sample_size)
    score1 = 2 + 5*z1 + (-3)*z2 + z3
    score2 = (score1 - min(score1))/max(score1 - min(score1))
    p = 1/(1 + np.exp((-1)*score1))
    x = bernoulli.rvs(p)

    y = 3 + 10*x + 3*z1 - 2*z2 + np.random.normal(0, 3, sample_size)

    plt.scatter(x=score2, y=p, c='dimgray', s=3, label="Probability")
    plt.scatter(x=score2, y=x, c='salmon', s=5, label="Intervention")
    plt.xlabel("XB, normalized")
    plt.legend()
    plt.savefig("../fig/logistic_plot.png", format="png")

    df = pd.DataFrame({"y": y,
                       # "x": pd.Series(x, dtype="category"),
                       "x": x,
                       "z1": z1,
                       "z2": z2,
                       "z3": z3})
    return df

def main():

    f = open('../result/estimation_ps.txt', 'w')

    df = gen_sample(sample_size, seed)

    mean_int = np.mean(df.y[df.x == 1])
    mean_ctr = np.mean(df.y[df.x == 0])

    # histogram
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4.5), dpi=80)
    ax.hist(df.y[df.x == 1], bins=20, normed=True,
            color="salmon", alpha=0.7, label="Intervention")
    ax.axvline(mean_int, color='r', linestyle='dashed')

    ax.hist(df.y[df.x == 0], bins=20, normed=True,
            color="lightgray", alpha=0.7, label="Control")
    ax.axvline(mean_ctr, color='r', linestyle='dashed')

    ax.legend()
    # plt.show()
    plt.savefig("../fig/histogram_target.png", format="png")

    # linear model
    print("----------simple OLS----------", file=f)
    model = ols("y ~ x", df).fit()
    print(model.summary(), file=f)

    # propensity score
    print("----------propensity score estimation----------", file=f)
    df = sm.add_constant(df, prepend=False)
    y = df.x
    X = df[['const', 'z1', 'z2', 'z3']]

    model_ps = glm(formula='x ~ z1 + z2 + z3',
                   data=df,
                   family=sm.families.Binomial()).fit()
    ps = pd.Series(model_ps.predict(X))
    print(model_ps.summary(), file=f)

    print("----------adjusted OLS----------", file=f)
    df['ps'] = ps
    model = ols('y ~ x + ps', data=df).fit()
    print(model.summary(), file=f)


    print("----------adjusted WLS----------", file=f)
    y = df.y
    X = df[['const', 'x', 'ps']]
    W = 1./df['ps']
    print(y.shape, X.shape, W.shape)
    print(W[:10])

    model = sm.WLS(y, X, weights=W).fit()
    print(model.summary(), file=f)

    print(df.head())
    mean_int = np.mean(df.y[(0.4 <= df.ps)*(df.ps < 0.5)*(df.x == 1)])
    mean_ctr = np.mean(df.y[(0.4 <= df.ps)*(df.ps < 0.5)*(df.x == 0)])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,4.5), dpi=80)
    ax1.hist(df.y[(0.4 <= df.ps)*(df.ps < 0.5)*(df.x == 1)], bins=20, normed=True,
            color="salmon", alpha=0.7, label="Intervention")
    ax1.axvline(mean_int, color='r', linestyle='dashed')
    ax1.hist(df.y[(0.4 <= df.ps)*(df.ps < 0.5)*(df.x == 0)], bins=20, normed=True,
            color="lightgray", alpha=0.7, label="Control")
    ax1.axvline(mean_ctr, color='r', linestyle='dashed')
    ax1.set_title("0.4 <= Propensity Score < 0.5")
    ax1.legend()

    mean_int = np.mean(df.y[(0.5 <= df.ps)*(df.ps < 0.6)*(df.x == 1)])
    mean_ctr = np.mean(df.y[(0.5 <= df.ps)*(df.ps < 0.6)*(df.x == 0)])
    ax2.hist(df.y[(0.5 <= df.ps)*(df.ps < 0.6)*(df.x == 1)], bins=20, normed=True,
            color="salmon", alpha=0.7, label="Intervention")
    ax2.axvline(mean_int, color='r', linestyle='dashed')
    ax2.hist(df.y[(0.5 <= df.ps)*(df.ps < 0.6)*(df.x == 0)], bins=20, normed=True,
            color="lightgray", alpha=0.7, label="Control")
    ax2.axvline(mean_ctr, color='r', linestyle='dashed')

    ax2.set_title("0.5 <= Propensity Score < 0.6")
    ax2.legend()

    # plt.show()
    plt.savefig("../fig/histogram_target_stratified.png", format="png")


if __name__ == "__main__":
    main()









