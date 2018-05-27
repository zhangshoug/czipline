"""
目标模块


"""
import cvxpy as cvx
import pandas as pd

from .utils import check_series_or_dict


class Objective(object):
    """
    Base class for objectives.
    """
    _objective = None
    _w = None          # 存储目标权重对象(变量)
    _asset_map = None  # 存储资产映射({Asset -> int})

    @property
    def objective(self):
        """
        使用属性保存问题目标对象(Minimize or Maximize)
        """
        return self._objective

    @property
    def w(self):
        return self._w

    @property
    def asset_map(self):
        return self._asset_map


class TargetWeights(Objective):
    """
    与投资组合目标权重最小距离的Objective

    Parameters
    ----------
        weights (pd.Series[Asset -> float] or dict[Asset -> float])
            资产与目标持有百分比的映射

    Notes
    -----
    一个目标值为1.0表示投资组合的当前净清算价值的100％在相应资产中应保持多头。

    一个目标值为-1.0表示投资组合的当前净清算价值的100％在相应资产中应保持空头。

    目标值为0.0的资产将被忽略，除非算法在给定资产中已有头寸。

    如果算法在资产中已有头寸，且没有提供目标权重，那么目标权重被假设为0.
    """

    def __init__(self, weights):
        check_series_or_dict(weights, 'weights')
        weights = pd.Series(weights)
        n = len(weights)
        w = cvx.Variable(shape=(n))
        # 更新属性
        self._w = w
        self._asset_map = {n:i for i, n in enumerate(weights.index)}
        self._objective = cvx.Minimize(cvx.sum_squares(w - weights.values))


class MaximizeAlpha(Objective):
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
        self._asset_map = {n:i for i, n in enumerate(alphas.index)}
        self._objective = cvx.Maximize(w.T * alphas)
