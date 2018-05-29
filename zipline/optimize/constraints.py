"""
限制模型

注意
1. volume：实质为成交额
"""

from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import pandas as pd
import numpy as np


class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop('w_bench', 0.)

    def weight_expr(self, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          w_plus: post-trade weights 交易后权重
          z: trade weights   交易权重
          v: portfolio value 投资组合价值
        """
        if w_plus is None:
            return [self._weight_expr(None, z, v)]
        return [self._weight_expr(w_plus - self.w_bench, z, v)]

    @abstractmethod
    def _weight_expr(self, w_plus, z, v):
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

    def _weight_expr(self, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          w_plus: 交易完成后的权重
          z: trade weights
          v: portfolio value
        """
        return [cvx.norm(cvx.abs(w_plus[:-1]), 1) <= self.max_]