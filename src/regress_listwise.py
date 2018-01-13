# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm

if __name__ == "__main__":
    df = pd.read_pickle('../data/sample_data.pkl')

    f = open('../result/regress_listwise.txt', 'w')

    print("----------complete data----------", file=f)
    X = df.loc[:,['x1','x2']].values
    y = df.y.values.reshape(-1, 1)
    model = sm.OLS(y, sm.add_constant(X))
    result = model.fit()
    print(result.summary(), file=f)

    print("independent-----------------------------------------------", file=f)
    d1 = df.loc[df.x1_case1 == 1]
    X = d1.loc[:,['x1','x2']].values
    y = d1.y.values.reshape(-1, 1)
    model = sm.OLS(y, sm.add_constant(X))
    result = model.fit()
    print(result.summary(), file=f)

    print("x2-dependent----------------------------------------------", file=f)
    d1 = df.loc[df.x1_case2 == 1]
    X = d1.loc[:,['x1','x2']].values
    y = d1.y.values.reshape(-1, 1)
    model = sm.OLS(y, sm.add_constant(X))
    result = model.fit()
    print(result.summary(), file=f)

    print("y-dependent-----------------------------------------------", file=f)
    d1 = df.loc[df.x1_case3 == 1]
    X = d1.loc[:, ['x1', 'x2']].values
    y = d1.y.values.reshape(-1, 1)
    model = sm.OLS(y, sm.add_constant(X))
    result = model.fit()
    print(result.summary(), file=f)





