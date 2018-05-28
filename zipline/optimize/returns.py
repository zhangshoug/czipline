"""
提供历史收益率数据，转换为优化计算所需的底层数据
"""


import cvxpy as cvx
from cvxportfolio.expression import Expression
from .utils.data_management import time_locator, null_checker

__all__ = ['ReturnsForecast', 'MPOReturnsForecast',
           'MultipleReturnsForecasts']


class BaseReturnsModel(Expression):
    pass


class ReturnsForecast(BaseReturnsModel):
    """A single return forecast.

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0., gamma_decay=None, name=None):
        null_checker(returns)
        self.returns = returns
        null_checker(delta)
        self.delta = delta
        self.gamma_decay = gamma_decay
        self.name = name

    def weight_expr(self, t, wplus, z=None, v=None):
        """
        估计alpha

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          alpha表达式
        """
        alpha = cvx.mul_elemwise(
            time_locator(self.returns, t, as_numpy=True), wplus)
        alpha -= cvx.mul_elemwise(
            time_locator(self.delta, t, as_numpy=True), cvx.abs(wplus))
        return cvx.sum_entries(alpha)

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: 持有权重表达式
          tau: 估计alpha时点

        Returns:
          An expression for the alpha.
        """

        alpha = self.weight_expr(t, wplus)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days**(-self.gamma_decay)
        return alpha


class MPOReturnsForecast(BaseReturnsModel):
    """A single alpha estimation.

    Attributes:
      alpha_data: A dict of series of return estimates.
    """

    def __init__(self, alpha_data):
        self.alpha_data = alpha_data

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        return self.alpha_data[(t, tau)].values.T * wplus


class MultipleReturnsForecasts(BaseReturnsModel):
    """A weighted combination of alpha sources.

    Attributes:
      alpha_sources: a list of alpha sources.
      weights: An array of weights for the alpha sources.
    """

    def __init__(self, alpha_sources, weights):
        self.alpha_sources = alpha_sources
        self.weights = weights

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr(t, wplus) * self.weights[idx]
        return alpha

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr_ahead(t,
                                              tau, wplus) * self.weights[idx]
        return alpha
