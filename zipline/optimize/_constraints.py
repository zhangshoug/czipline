"""
优化限定模块

限定权重分多头与空头，输入为字典或序列，如果以字典形式输入，为
但内部转换为多头与空头权重
"""


import pandas as pd
import cvxpy as cvx

from .utils import check_series_or_dict


class Constraint(object):
    """
    Base class for constraints.
    """

    def make_constraints(self, w, asset_maps=None):
        # 生成限制列表(返回列表)
        raise NotImplementedError('需要在子类实现')


class MaxGrossExposure(Constraint):
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

    def __init__(self, max_):
        self.max_ = max_

    def make_constraints(self, w):
        return [cvx.sum(cvx.abs(w)) <= self.max_]


class NetExposure(Constraint):
    """
    对投资组合净敞口的约束

    要求投资组合中所有资产的权重总和（正值或负值）在`min_`和`max_`之间。

    Parameters
    ----------
        min_ :float
            投资组合最小净敞口
        max_ : float
            投资组合最小净敞口

    Examples
    --------
    将投资组合的多头和空头之间的价值差异限制在当前投资组合价值的-10％和10％之间
    >>> NetExposure(-0.1, 0.1)
    """

    def __init__(self, min_, max_):
        assert min_ < max_, '输入的最小值应小于最大值'
        self.min_ = min_
        self.max_ = max_

    def make_constraints(self, w):
        return [
            w >= self.min_,
            w <= self.max_,
        ]

class DollarNeutral(Constraint):
    """
    限定多头与空头风险敞口必须等规模

    要求投资组合中所有资产的权重总和（正值或负值）落在容差区间

    Parameters
    ----------
    tolerance : float, optional
        允许的净风险敞口偏移率。默认值是0.0001
    """

    def __init__(self, tolerance=0.0001):
        self.tolerance = tolerance


class NetGroupExposure(Constraint):
    """
    对各组资产的净风险敞口约束边界

    组由（资产 -> 标签）映射来定义。每个唯一标签生成一个约束，限制该标签所有资
    产权重之和应该落在下限和上限之间。

    用（标签 - >浮点数）映射定义最小/最大组风险敞口。
    常见组标签的例子是部门，行业和国家。

    Parameters
    ----------
    labels (pd.Series[Asset -> object] or dict[Asset -> object])
        Map from asset -> group label.
    min_weights (pd.Series[object -> float] or dict[object -> float])
        Map from group label to minimum net exposure to assets in that group.
    max_weights (pd.Series[object -> float] or dict[object -> float])
        Map from group label to maximum net exposure to assets in that group.
    etf_lookthru (pd.DataFrame, optional)
        Indexed by constituent assets x ETFs, expresses the weight of each 
        constituent in each ETF.

        A DataFrame containing ETF constituents data. Each column of the frame 
        should contain weights (from 0.0 to 1.0) representing the holdings for 
        an ETF. Each row should contain weights for a single stock in each ETF. 
        Columns should sum approximately to 1.0. If supplied, ETF holdings in 
        the current and target portfolio will be decomposed into their constituents 
        before constraints are applied.

    Examples
    --------
    假设我们对四种股票感兴趣：AAPL，MSFT，TSLA和GM。
 
    AAPL和MSFT在技术领域。TSLA和GM都处于消费者周期性行业。希望我们的投资组合中
    不超过50％投入科技行业，不超过25％投入到消费品周期性行业。

    可以构建一个表达上述偏好的约束条件如下：
    # Map from each asset to its sector.
    labels = {AAPL: 'TECH', MSFT: 'TECH', TSLA: 'CC', GM: 'CC'}

    # Maps from each sector to its min/max exposure.
    min_exposures = {'TECH': -0.5, 'CC': -0.25}
    max_exposures = {'TECH': 0.5, 'CC': 0.5}

    constraint = NetGroupExposure(labels, min_exposures, max_exposures)

    Notes
    -----
    对于不应设置最低限制的组，请设置：

    min_weights[group_label] = opt.NotConstrained

    对于不应设置最高限制的组，请设置：

    max_weights[group_label] = opt.NotConstrained
    """

    def __init__(self, labels, min_weights, max_weights, etf_lookthru=None):
        check_series_or_dict(labels, 'labels')
        check_series_or_dict(min_weights, 'min_weights')
        check_series_or_dict(max_weights, 'max_weights')
        self.labels = labels
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.etf_lookthru = etf_lookthru

    @classmethod
    def with_equal_bounds(labels, min_, max_, etf_lookthru=None):
        """
        Special case constructor that applies static lower and upper bounds 
        to all groups.

        NetGroupExposure.with_equal_bounds(labels, min, max)

        is equivalent to:

        NetGroupExposure(
            labels=labels,
            min_weights=pd.Series(index=labels.unique(), data=min),
            max_weights=pd.Series(index=labels.unique(), data=max),
        )

        Parameters
        ----------
            labels：(pd.Series[Asset -> object] or dict[Asset -> object])
                Map from asset -> group label.
            min：float
                Lower bound for exposure to any group.
            max：float
                Upper bound for exposure to any group.
        """
        return NetGroupExposure(labels, min_, max_, etf_lookthru)


class PositionConcentration(Constraint):
    """
    约束强制执行最小/最大头寸权重

    Parameters
    ----------  
    min_weights (pd.Series[Asset -> float] or dict[Asset -> float])
        Map from asset to minimum position weight for that asset.
    max_weights (pd.Series[Asset -> float] or dict[Asset -> float])
        Map from asset to maximum position weight for that asset.
    default_min_weight (float, optional)
        Value to use as a lower bound for assets not found in min_weights. 
        Default is 0.0.
    default_max_weight (float, optional)
        Value to use as a lower bound for assets not found in max_weights. 
        Default is 0.0.
    etf_lookthru (pd.DataFrame, optional)
        Indexed by constituent assets x ETFs, expresses the weight of each constituent 
        in each ETF. 
        A DataFrame containing ETF constituents data. Each column of the frame should 
        contain weights (from 0.0 to 1.0) representing the holdings for an ETF. Each row 
        should contain weights for a single stock in each ETF. Columns should sum 
        approximately to 1.0. If supplied, ETF holdings in the current and target 
        portfolio will be decomposed into their constituents before constraints are 
        applied. 
    
    Notes   
    -----
    负的权重值被解释为限制空头头寸的数量。
    最小权重为0.0会将资产限制为仅为多头。
    最大权重为0.0会将资产限制为仅为空头。
    一个常见的特例是创建一个`PositionConcentration`约束，该约束对所有资产应用共享的下限/上限。 
    另一个构造函数`PositionConcentration.with_equal_bounds()`提供了一个更简单的API来支持这种用例。
  
    classmethod with_equal_bounds(min, max, etf_lookthru=None)  
        Special case constructor that applies static lower and upper bounds to all assets.  
        PositionConcentration.with_equal_bounds(min, max)   
        is equivalent to:   
        PositionConcentration(pd.Series(), pd.Series(), min, max)   
        Parameters:	    
            min (float) – Minimum position weight for all assets.
            max (float) – Maximum position weight for all assets.   
    """

    def __init__(self,
                 min_weights,
                 max_weights,
                 default_min_weight=0.0,
                 default_max_weight=0.0,
                 etf_lookthru=None):
        check_series_or_dict(min_weights, 'min_weights')
        check_series_or_dict(max_weights, 'max_weights')
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.default_min_weight = default_min_weight
        self.default_max_weight = default_max_weight
        self.etf_lookthru = etf_lookthru

    @classmethod
    def with_equal_bounds(min_, max_, etf_lookthru=None):
        """
        Special case constructor that applies static lower and upper bounds to all assets.  
        PositionConcentration.with_equal_bounds(min, max)   
        is equivalent to:   
        PositionConcentration(pd.Series(), pd.Series(), min, max)   
        Parameters:	    
            min_ (float) – Minimum position weight for all assets.
            max_ (float) – Maximum position weight for all assets.   
        """
        return PositionConcentration(pd.Series(), pd.Series(), min_, max_)


class FactorExposure(Constraint):
    """
    对风险因子集合限制净风险敞口

    因子载入被指定为列为因子标签和其索引为资产的浮点数DataFrame。
    最小和最大因子净风险敞口由因子标签与最小/最大映射来指定。
    对于加载数据框的每列，限定：
    (new_weights * loadings[column]).sum() >= min_exposure[column]
    (new_weights * loadings[column]).sum() <= max_exposure[column]

    Parameters
    ----------
    loadings (pd.DataFrame)
        An (assets x labels) frame of weights for each (asset, factor) pair.
    min_exposures (dict or pd.Series)
        Minimum net exposure values for each factor.
    max_exposures (dict or pd.Series)
        Maximum net exposure values for each factor.
    """

    def __init__(self, loadings, min_exposures, max_exposures):
        check_series_or_dict(min_exposures, 'min_exposures')
        check_series_or_dict(max_exposures, 'max_exposures')
        self.loadings = loadings
        self.min_exposures = min_exposures
        self.max_exposures = max_exposures


class Pair(Constraint):
    """
    表示一对反向权重股票的约束

    Parameters
    ----------
    long (Asset)
        The asset to long.
    short (Asset)
        The asset to short.
    hedge_ratio (float, optional)
        多头与空头权重绝对值比率。 要求大于0.默认值为1.0，表示权重相等。
    tolerance (float, optional)
        允许计算权重的套期保值比率与给定套期保值比率在任一方向上不同。要求大于或等于0.默认值为0.0。
    """

    def __init__(self, long, short, hedge_ratio=1.0, tolerance=0.0):
        self.long = long
        self.short = short
        self.hedge_ratio = hedge_ratio
        self.tolerance = tolerance


class Basket(Constraint):
    """
    限制要求对一篮子股票进行有限的净风险敞口

    Parameters
    ----------
    assets (iterable[Asset])
        Assets to be constrained.
    min_net_exposure (float)
        Minimum allowed net exposure to the basket.
    max_net_exposure (float)
        Maximum allowed net exposure to the basket.
    """

    def __init__(self, assets, min_net_exposure, max_net_exposure):
        self.assets = assets
        self.min_net_exposure = min_net_exposure
        self.max_net_exposure = max_net_exposure


class Frozen(Constraint):
    """
    限定资产头寸不得更改

    Parameters
    ----------
    asset_or_assets : Asset or sequence[Asset]
        Asset(s) whose weight(s) cannot change.
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display


class ReduceOnly(Constraint):
    """
    限制资产的权重只能减少且不得低于零
    
    Parameters
    ----------
    asset：Asset
        不得增加头寸的资产
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display


class LongOnly(Constraint):
    """
    限定资产不得持有空头头寸

    Parameters
    ----------
    asset_or_assets：Asset or iterable[Asset]
        资产头寸必须为多头或为0
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display


class ShortOnly(Constraint):
    """
    限定资产不得持有多头头寸
    Parameters
    ----------
    asset_or_assets：Asset or iterable[Asset]
        资产头寸必须为空头或为0
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display


class FixedWeight(Constraint):
    """
    限定资产的权重必须为特定权重

    Parameters
    ----------
    asset：Asset
        资产
    weight：float
        特定权重值
    """

    def __init__(self, asset, weight):
        self.asset = asset
        self.weight = weight


class CannotHold(Constraint):
    """
    限定资产头寸规模必须为0

    Parameters
    ----------
    asset_or_assets：Asset or iterable[Asset]
        不得持有头寸的资产或多个资产
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display