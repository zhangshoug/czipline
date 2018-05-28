"""
限制模型

注意
1. volume：实质为成交额
"""

from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd
import numpy as np

from .risks import locator

__all__ = ['LongOnly', 'LeverageLimit', 'LongCash', 'MaxTrade']


class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop('w_bench', 0.)

    def weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: post-trade weights 交易后权重
          z: trade weights   交易权重
          v: portfolio value 投资组合价值
        """
        if w_plus is None:
            return self._weight_expr(t, None, z, v)
        return self._weight_expr(t, w_plus - self.w_bench, z, v)

    @abstractmethod
    def _weight_expr(self, t, w_plus, z, v):
        pass


class MaxGrossExposure(BaseConstraint):
    """
    限制投资组合的最大总风险敞口

    要求投资组合权重的绝对值之和小于`max_`

    Parameters
    ----------
    max_ : float
        投资组合的最大总风险敞口

    Examples
    --------
    # 将投资组合的多头和空头的总价值限制在目前投资组合价值的1.5倍以下
    >>> MaxGrossExposure(1.5)
    """

    def __init__(self, max_, **kwargs):
        self.max_ = max_
        super(MaxGrossExposure, self).__init__(**kwargs)

    def make_constraints(self, w):
        return [cvx.sum(cvx.abs(w)) <= self.max_]

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: 交易完成后的权重
          z: trade weights
          v: portfolio value
        """
        return cvx.abs(z[:-1]) * v <= \
            np.array(locator(self.ADVs, t)) * self.max_fraction


class MaxTrade(BaseConstraint):
    """限制最大交易量

    ADVs：pd.DataFrame(index:日期， 列：资产) 股票成交量数据
    max_fraction：相对于成交量允许交易的最大比例
    """

    def __init__(self, ADVs, max_fraction=0.05, **kwargs):
        # 传入历史成交额数据
        self.ADVs = ADVs
        self.max_fraction = max_fraction
        super(MaxTrade, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: 交易后权重
          z: 交易权重
          v: 投资组合价值
        """
        #
        return cvx.abs(z[:-1]) * v <= \
            np.array(locator(self.ADVs, t)) * self.max_fraction

        # TODO fix the locator for this usecase


class LongOnly(BaseConstraint):
    """A long only constraint. 只允许多头头寸
    """

    def __init__(self, **kwargs):
        super(LongOnly, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          wplus: holdings
        """
        return w_plus >= 0


class LeverageLimit(BaseConstraint):
    """交易杠杆限制

    Attributes:
      limit: 一个序列或数字给定杠杆限制
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(LeverageLimit, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          wplus: holdings
        """
        if isinstance(self.limit, pd.Series):
            limit = self.limit.loc[t]
        else:
            limit = self.limit
        return cvx.norm(w_plus, 1) <= limit


class LongCash(BaseConstraint):
    """Requires that cash be non-negative. 现金非负限制
    """

    def __init__(self, **kwargs):
        super(LongCash, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          wplus: holdings
        """
        return w_plus[-1] >= 0
