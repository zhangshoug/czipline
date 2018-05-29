import pandas as pd
import cvxpy as cvx

stocks = ['000333', '300001', '600771', '000002', '600645']

target = pd.Series([0.2, 0.3, 0.1, 0.2, 0.2], index=stocks)

n = len(target)
w = cvx.Variable(n)

prob = cvx.Problem(
    cvx.Minimize(cvx.norm(cvx.abs(w), 1)), [cvx.sum_entries(w) == 1, w >= 0])
prob.solve()
