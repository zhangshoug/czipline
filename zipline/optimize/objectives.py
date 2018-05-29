"""
目标模块
"""
from abc import ABCMeta, abstractmethod

import cvxpy as cvx
import logbook
import numpy as np
import pandas as pd

from .utils import check_series_or_dict
from .utils.data_management import null_checker, time_locator


class ObjectiveBase(object):
    """ Base class for a trading policy. """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.constraints = []

    @abstractmethod
    def objective(self, portfolio):
        """
        给定当前投资组合及时间t的交易列表
        Trades list given current portfolio and time t.
        """
        return NotImplemented


class BaseRebalance(ObjectiveBase):
    def _rebalance(self, portfolio):
        return sum(portfolio) * self.target - portfolio


class TargetWeights(BaseRebalance):
    """
    与投资组合目标权重最小距离的ObjectiveBase

    Parameters
    ----------
    target：pd.Series[Asset -> float] or dict[Asset -> float]
        资产目标权重(或资产目标权重映射)

    Notes
    -----
    一个目标值为1.0表示投资组合的当前净清算价值的100％在相应资产中应保持多头。

    一个目标值为-1.0表示投资组合的当前净清算价值的100％在相应资产中应保持空头。

    目标值为0.0的资产将被忽略，除非算法在给定资产中已有头寸。

    如果算法在资产中已有头寸，且没有提供目标权重，那么目标权重被假设为0.
    """

    def __init__(self, target):
        check_series_or_dict(target, 'target')
        self.target = pd.Series(target)
        self.tracking_error = 1e-4
        super(TargetWeights, self).__init__()


    def objective(self, portfolio):
        if portfolio is None:
            # 此时返回的是权重 TODO：注意一致性，应返回要成交的金额
            return self.target
        weights = portfolio / sum(portfolio)
        diff = (weights - self.target).values

        if np.linalg.norm(diff, 2) > self.tracking_error:
            return [self._rebalance(portfolio)]
        else:
            return [self._nulltrade(portfolio)]


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
        alphas = pd.Series(alphas)
        n = len(alphas)
        w = cvx.Variable(shape=(n, 1))

        # 更新属性
        self._w = w
        self._asset_map = {n: i for i, n in enumerate(alphas.index)}
        self._ObjectiveBase = cvx.Maximize(w.T * alphas)
