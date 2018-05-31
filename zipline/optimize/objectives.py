"""
目标模块
"""
from abc import ABCMeta, abstractmethod

import logbook
import numpy as np
import pandas as pd

import cvxpy as cvx

from .utils import check_series_or_dict

__all__ = ['TargetWeights', 'MaximizeAlpha']


class ObjectiveBase(object):
    """目标基类"""
    __metaclass__ = ABCMeta

    cash_key = 'cash'

    @abstractmethod
    def _expr(self):
        """目标"""
        return NotImplemented

    def make_weight(self, n, assets):
        """构造多、空权重双变量"""
        self.n = len(assets)
        # 多头权重(数字非负)
        self.long_w = cvx.Variable(n, nonneg=True)
        # 空头权重(数字非正)
        self.short_w = cvx.Variable(n, nonpos=True)
        # 绝对值权重
        self.w = self.long_w - self.short_w
        # 资产索引(用于调整限定条件多空权重变量参数位置)
        self.assets = assets

    @property
    def weights_series(self):
        """绝对值权重表达式序列"""
        return pd.Series([self.w[i] for i in range(self.n)], index=self.assets)

    @property
    def weights_value(self):
        """绝对值权重值"""
        return self.weights_series.map(lambda x: round(x.value, 4))

    @property
    def long_weights_series(self):
        """多头权重表达式序列"""
        return pd.Series(
            [self.long_w[i] for i in range(self.n)], index=self.assets)

    @property
    def long_weights_value(self):
        """多头权重值"""
        return self.long_weights_series.map(lambda x: round(x.value, 4))

    @property
    def short_weights_series(self):
        """空头权重表达式序列"""
        return pd.Series(
            [self.short_w[i] for i in range(self.n)], index=self.assets)

    @property
    def short_weights_value(self):
        """空头权重值"""
        return self.short_weights_series.map(lambda x: round(x.value, 4))


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
        self.weights = pd.Series(weights)
        n = len(weights)
        # TODO:暂时没有考虑cash
        self.make_weight(n, self.weights.index)
        super(TargetWeights, self).__init__()

    def _expr(self):
        # 目标权重
        t_w = self.weights.values
        # 必须完全复制值
        l_w, s_w = np.array(t_w), np.array(t_w)
        l_w[t_w <= 0] = 0.
        s_w[t_w >= 0] = 0.
        long_err = cvx.sum_squares(self.long_w - l_w)
        short_err = cvx.sum_squares(self.short_w - s_w)
        return long_err + short_err

    @property
    def objective(self):
        return cvx.Minimize(self._expr())


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
        self.alphas = pd.Series(alphas)
        n = len(alphas)
        # TODO:暂时没有考虑cash
        self.make_weight(n, self.alphas.index)
        super(MaximizeAlpha, self).__init__()

    def _expr(self):
        alphas = self.alphas.values
        # # 必须完全复制值
        long_profit = alphas.T * self.long_w  # 多头加权收益
        short_profit = alphas.T * self.short_w  # 空头加权收益
        return long_profit + short_profit

    @property
    def objective(self):
        return cvx.Maximize(self._expr())
