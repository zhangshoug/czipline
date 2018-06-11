"""
目标模块

假设前提：
    1. 无论是目标权重或当前权重，在特定时刻，单个资产不得同时存在多头与空头权重；
"""

from abc import ABCMeta, abstractmethod

import logbook
import numpy as np
import pandas as pd

import cvxpy as cvx

from .utils import check_series_or_dict, get_ix

__all__ = ['ObjectiveBase', 'TargetWeights', 'MaximizeAlpha']


class ObjectiveBase(object):
    """目标基类。子类objective属性代表cvxpy目标对象"""
    __metaclass__ = ABCMeta

    def __init__(self, digits=6):
        self.cash_key = 'cash'
        self.digits = digits

    def __repr__(self):
        return "%s(n=%s)" % (self.__class__.__name__, len(self._new_index))

    def _make_variable(self, current):
        if current is None:
            self._new_index = self._old_index
        else:
            self._new_index = self._old_index.union(current.index).unique()
        n = len(self._new_index)
        self._new_weights = cvx.Variable(n, name='new_weights')

    @property
    def new_weights(self):
        """权重表达式"""
        return self._new_weights

    @property
    def new_weights_series(self):
        """权重表达式序列"""
        n = len(self._new_index)
        return pd.Series(
            [self._new_weights[i] for i in range(n)], index=self._new_index)

    @property
    def new_weights_value(self):
        """多头权重值"""
        return self.new_weights_series.map(
            lambda x: round(x.value, self.digits))


class TargetWeights(ObjectiveBase):
    """
    与投资组合目标权重最小距离的ObjectiveBase

    Parameters
    ----------
    weights：pd.Series[Asset -> float] or dict[Asset -> float]
        资产目标权重(或资产目标权重映射)

    Notes
    -----
    一个目标值为1.0表示投资组合的当前净清算价值的100％在相应资产中应保持多头。

    一个目标值为-1.0表示投资组合的当前净清算价值的100％在相应资产中应保持空头。

    目标值为0.0的资产将被忽略，除非算法在给定资产中已有头寸。

    如果算法在资产中已有头寸，且没有提供目标权重，那么目标权重被假设为0.
    """

    def __init__(self, weights):
        check_series_or_dict(weights, 'weights')
        # 目标权重不得包含Nan值
        self._target_weights = pd.Series(weights).fillna(0.0)
        self._old_index = self._target_weights.index
        super(TargetWeights, self).__init__()

    def to_cvxpy(self, current_weights):
        self._make_variable(current_weights)
        self._target_weights = self._target_weights.reindex(
            self._new_index).fillna(0)
        err = cvx.sum_squares(self._new_weights - self._target_weights.values)
        return cvx.Minimize(err)


class MaximizeAlpha(ObjectiveBase):
    """
    对于一个alpha向量最大化weights.dot(alphas)的目标对象

    理想情况下，资产alpha系数应使得期望值与目标投资组合持有的时间成正比。
    
    特殊情况下，alphas只是每项资产的期望收益估计值，这个目标只是最大化整个投资
    组合的预期收益。

    Parameters
    ----------
    alphas ： pd.Series[Asset -> float] 或 dict[Asset -> float]
        资产与这些资产的α系数之间的映射

    Notes
    ------

    这个目标总是与`MaxGrossExposure`约束一起使用，并且通常应该与`PositionConcentration`
    约束一起使用。

    如果没有对总风险的限制，这个目标会产生一个错误，试图为每个非零的alphas系数的资产分配无限的资本。

    如果没有对单个头寸大小的限制，这个目标将分配所有的头寸到预期收益最大的单项资产中。
    """

    def __init__(self, alphas):
        check_series_or_dict(alphas, 'alphas')
        # 因子值不得为Nan，以0代替
        self._alphas = pd.Series(alphas).fillna(0.0)
        self._old_index = self._alphas.index
        super(MaximizeAlpha, self).__init__()

    def to_cvxpy(self, current_weights):
        self._make_variable(current_weights)
        self._alphas = self._alphas.reindex(self._new_index).fillna(0)
        profit = self._alphas.values.T * self._new_weights  # 加权收益
        return cvx.Maximize(cvx.sum(profit))
