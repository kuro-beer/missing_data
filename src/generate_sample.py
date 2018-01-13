# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

missing_ratio = 0.3
sample_size = 200
seed = 123

if __name__ == "__main__":
    np.random.seed(seed)
    x1 = np.random.randn(sample_size)
    x2 = np.random.randn(sample_size)
    y = 5 + 3*x1 - 2*x2 + np.random.normal(0, 4, sample_size)

    df = pd.DataFrame({"y": y,
                       "x1": x1,
                       "x2": x2})

    df['x1_case1'] = df.x1.map(
        lambda x: 1 if np.random.rand() >= missing_ratio else 0
    )
    df['x1_case2'] = (df.x2 >= norm.ppf(missing_ratio)).map(
        lambda x: np.int(x)
    )
    df['x1_case3'] = (df.y >= df.y.quantile(missing_ratio)).map(
        lambda x: np.int(x)
    )
    print(df.head(10))

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2,
                                                     ncols=2,
                                                     figsize=(9,9),
                                                     dpi=80)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    xvar = df.x1
    yvar = df.y
    ax11.plot(xvar, yvar, 'o',
              color='navy',
              markerfacecolor='w',
              alpha=0.7)
    ax11.set_title('complete data')
    ax11.set_xlabel('x1')
    ax11.set_ylabel('y')
    ax11.set_xlim(-3, 3)
    ax11.set_ylim(-10, 20)

    xvar = df.x1.loc[df.x1_case1 == 1]
    yvar = df.y.loc[df.x1_case1 == 1]
    ax12.plot(xvar, yvar, 'o',
              color='navy',
              markerfacecolor='w',
              alpha=0.7)
    ax12.set_title('independent missing')
    ax12.set_xlabel('x1')
    ax12.set_ylabel('y')
    ax12.set_xlim(-3, 3)
    ax12.set_ylim(-10, 20)

    xvar = df.x1.loc[df.x1_case2 == 1]
    yvar = df.x2.loc[df.x1_case2 == 1]
    ax21.plot(xvar, yvar, 'o',
              color='darkgreen',
              markerfacecolor='w',
              alpha=0.7)
    ax21.set_title('x2-dependent missing')
    ax21.set_xlabel('x1')
    ax21.set_ylabel('x2', color="green")
    ax21.set_xlim(-3, 3)
    ax21.set_ylim(-3, 3)
    ax21.hlines([norm.ppf(missing_ratio)], -3, 3, "red", linestyles='dashed')

    xvar = df.x1.loc[df.x1_case3 == 1]
    yvar = df.y.loc[df.x1_case3 == 1]
    ax22.plot(xvar, yvar, 'o',
              color='navy',
              markerfacecolor='w',
              alpha=0.7)
    ax22.set_title('y-dependent missing')
    ax22.set_xlabel('x1')
    ax22.set_ylabel('y')
    ax22.set_xlim(-3, 3)
    ax22.set_ylim(-10, 20)
    ax22.hlines([df.y.quantile(missing_ratio)], -10, 20, "red", linestyles='dashed')

    plt.savefig('../fig/sample_scatter.png', format='png')
    df.to_pickle('../data/sample_data.pkl')




