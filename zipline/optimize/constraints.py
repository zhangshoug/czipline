"""
限制模型
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from collections import Iterable
import cvxpy as cvx

from .utils import check_series_or_dict, get_ix

__all__ = [
    'NotConstrained',
    'MaxGrossExposure',
    'NetExposure',
    'NetGroupExposure',
    'DollarNeutral',
    'NetGroupExposure',
    'PositionConcentration',
    'FactorExposure',
    'Pair',
    'Basket',
    'Frozen',
    'ReduceOnly',
    'LongOnly',
    'ShortOnly',
    'FixedWeight',
    'CannotHold',
    'NotExceed',
    'NotLessThan',
]

NotConstrained = 'NotConstrained'


class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.constraint_assets = None

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s()" % (self.__class__.__name__)

    def gen_constraints(self,
                        weight_var,
                        weight_var_series=None,
                        init_weights=None):
        """
        返回限制列表

        Args
        ----
        weight_var:权重变量
        weight_var_series:权重表达式序列(以assets为索引)
        init_weights:初始权重值序列(以assets为索引)
        """
        return self.to_cvxpy(weight_var, weight_var_series,
                                      init_weights)

    @abstractmethod
    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        """子类完成具体表达式"""
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

    def __init__(self, max_):
        self.max_ = max_
        super(MaxGrossExposure, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s)" % (self.__class__.__name__, self.max_)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        return [cvx.sum(cvx.abs(weight_var)) <= self.max_]
        # return [cvx.norm(cvx.abs(weight_var), 2) <= self.max_]


class NetExposure(BaseConstraint):
    """
    对投资组合净敞口的约束

    要求权重合计(正负权重自然累加)介于`min_`和`max_`之间。

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

    注
    --
        1. 一般与`MaxGrossExposure`结合使用，否则会无上界
    """

    def __init__(self, min_, max_):
        assert min_ < max_, '输入的最小值应小于最大值'
        self.min_ = min_
        self.max_ = max_
        super(NetExposure, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s,%s)" % (self.__class__.__name__, self.min_, self.max_)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        total = cvx.sum(weight_var)
        return [total >= self.min_, total <= self.max_]


class DollarNeutral(BaseConstraint):
    """
    限定多头与空头风险敞口必须等规模

    要求投资组合中多头权重累计与空头权重累计之差落在容差区间

    Parameters
    ----------
    tolerance : float, optional
        允许的净风险敞口偏移率。默认值是0.0001

    注
    --
        其逻辑是，不需要维持原权重，只需在新的权重中对涉及到的股票多空偏差落在允许范围即可
        只限定初始权重多空偏差

        如果只是限定新的目标权重多空同等规模，设定`NetExposure(-0.0001,0.0001)`有同样效果，
        无需多此一举。初步判断逻辑如上所述。
    """

    def __init__(self, tolerance=0.0001):
        # 不应限制
        # assert tolerance < 0.01, 'tolerance太大没有意义'
        self.tolerance = tolerance
        super(DollarNeutral, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s)" % (self.__class__.__name__, self.tolerance)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        # 在原权重中分别找出多头与空头权重序列
        long_w = init_weights[init_weights >= 0]
        short_w = init_weights[init_weights < 0]

        # 为使shape一致，需要使用fillna，以0值填充
        common_index = long_w.index.union(short_w.index).unique()
        # 与其值无关
        # long_w = long_w.reindex(common_index).fillna(0)
        # short_w = short_w.reindex(common_index).fillna(0)

        # 找到涉及到变量的位置
        ix = get_ix(weight_var_series, common_index)

        # 限定其和的绝对值不超过容忍阀值
        return [
            cvx.abs(cvx.sum(weight_var[ix])) <= self.tolerance,
        ]


class NetGroupExposure(BaseConstraint):
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

        # 暂未使用该参数

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
    1. 对于不应设置最低限制的组，请设置：

    >>> min_weights[group_label] = opt.NotConstrained

    2. 对于不应设置最高限制的组，请设置：

    >>> max_weights[group_label] = opt.NotConstrained

    3. 如果组最小限制大于0，组内依然允许空头，只需保持净敞口大于0

    4. 同样，如果组最大限制小于0，组内依然允许多头，只需保持净敞口小于0
    """

    def __init__(self, labels, min_weights, max_weights, etf_lookthru=None):
        # msg = """
        # 每组都必须指定上下限。如果某组不需要上限或下限，请使用：
        # min_weights[group_label] = opt.NotConstrained 或
        # max_weights[group_label] = opt.NotConstrained
        # """
        check_series_or_dict(labels, 'labels')
        check_series_or_dict(min_weights, 'min_weights')
        check_series_or_dict(max_weights, 'max_weights')
        labels = pd.Series(labels)
        min_weights = pd.Series(min_weights)
        max_weights = pd.Series(max_weights)
        # 现在可以不对称限定组
        # assert min_weights.index.equals(max_weights.index), msg
        self.labels = labels
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.etf_lookthru = etf_lookthru
        # 重写基类的constraint_assets属性
        self.constraint_assets = labels.index
        super(NetGroupExposure, self).__init__()

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
        # 依然要先检查类型是否正确(主要要使用unique方法)
        check_series_or_dict(labels, 'labels')
        labels = pd.Series(labels)
        return NetGroupExposure(
            labels=labels,
            min_weights=pd.Series(index=labels.unique(), data=min_),
            max_weights=pd.Series(index=labels.unique(), data=max_),
        )

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        constraints = []
        labels = self.labels
        groups = self.max_weights.index.union(self.min_weights.index).unique()

        msg_fmt = '组"{}"最小值"{}"应小于等于最大值"{}"'
        for g in groups:
            min_ = self.min_weights[g]
            max_ = self.max_weights[g]
            # 只有二方都限定，才比较
            limited_min = min_ != 'NotConstrained'
            limited_max = max_ != 'NotConstrained'
            if limited_min and limited_max:
                assert min_ <= max_, msg_fmt.format(g, min_, max_)
            # 该组涉及到的股票
            assets = labels[labels == g].index
            # 股票对应的变量位置序号
            ix = get_ix(weight_var_series, assets)

            # 如果该组没有对应序号，跳过
            if len(ix) == 0:
                continue

            var_list = weight_var[ix]

            # 如果限定该组上限
            # if limited_max and len(var_list):
            if limited_max:
                constraints.append(cvx.sum(var_list) <= max_)

            # 如果限定该组下限
            # if limited_min  and len(var_list):
            if limited_min:
                constraints.append(cvx.sum(var_list) >= min_)
        return constraints


class PositionConcentration(BaseConstraint):
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
        assert default_min_weight <= default_max_weight, '默然最小值必须小于等于默认最大值'
        check_series_or_dict(min_weights, 'min_weights')
        check_series_or_dict(max_weights, 'max_weights')
        min_weights = pd.Series(min_weights)
        max_weights = pd.Series(max_weights)
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.default_min_weight = default_min_weight
        self.default_max_weight = default_max_weight
        self.etf_lookthru = etf_lookthru

        # 重写基类的constraint_assets属性
        assets = min_weights.index.union(max_weights.index).unique()
        self.constraint_assets = assets
        super(PositionConcentration, self).__init__()

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

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        # 限定条件表达式列表
        constraints = []
        d_min = self.default_min_weight
        d_max = self.default_max_weight
        # 逻辑关系：如果提供部分限制，所涉及到的股票视为总体
        # 总体中没有明确最小或最大限制，使用默认值
        # 如果全部使用默认值，总体为期初权重序列的Index
        if len(self.min_weights) == 0 and len(self.max_weights) == 0:
            msg = '全部股票限定使用默认最小限制与最大限制时，初始权重不得为None'
            assert init_weights is not None, msg
            self.min_weights = pd.Series(d_min, index=init_weights.index)
            self.max_weights = pd.Series(d_max, index=init_weights.index)
        # 共同股票
        common_index = self.min_weights.index.union(
            self.max_weights.index).unique()

        candidates_min = self.min_weights.reindex(common_index).fillna(d_min)
        candidates_max = self.max_weights.reindex(common_index).fillna(d_max)

        if not all(candidates_max >= candidates_min):
            err_values = common_index[candidates_max < candidates_min]
            msg_fmt = '在限制：{}中, 股票：{} 设置最小权重为：{}，最大权重为：{}，最小值大于最大值'
            for asset in err_values:
                print(
                    msg_fmt.format(
                        str(self), asset, candidates_min[asset],
                        candidates_max[asset]))
            raise ValueError('单个股票的最大值应大于等于最小值')

        # 强制可迭代
        if len(common_index) == 1:
            common_index = [common_index]
        # 指定限制在权重变量中所对应的位置序号
        locs = get_ix(weight_var_series, common_index)
        for loc, asset in zip(locs, common_index):
            min_ = candidates_min[asset]
            max_ = candidates_max[asset]
            w_var = weight_var[loc]
            constraints.extend([w_var >= min_, w_var <= max_])
        return constraints


class FactorExposure(BaseConstraint):
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
        self.min_exposures = pd.Series(min_exposures)
        self.max_exposures = pd.Series(max_exposures)

        # 重写基类的constraint_assets属性
        self.constraint_assets = loadings.index
        super(FactorExposure, self).__init__()

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        min_cons, max_cons = [], []
        loadings = self.loadings  #.loc[weight_var_series.index].dropna()
        ix = get_ix(weight_var_series, self.loadings.index)
        for c in loadings.columns:
            min_ = self.min_exposures[c]
            max_ = self.max_exposures[c]
            assert min_ <= max_, '{} 列{}最小值"{}"必须小于等于最大值"{}"'.format(
                str(self), c, min_, max_)
            min_cons.append(cvx.sum(weight_var[ix] * loadings[c]) >= min_)
            max_cons.append(cvx.sum(weight_var[ix] * loadings[c]) <= max_)
        return min_cons + max_cons


class Pair(BaseConstraint):
    """
    表示一对反向权重股票的约束

    Parameters
    ----------
    long_ (Asset)
        The asset to long.
    short_ (Asset)
        The asset to short.
    hedge_ratio (float, optional)
        多头与空头权重绝对值比率。 要求大于0.默认值为1.0，表示权重相等。
    tolerance (float, optional)
        允许计算权重的套期保值比率与给定套期保值比率在任一方向上不同。要求大于或等于0.默认值为0.0。
    """

    def __init__(self, long_, short_, hedge_ratio=1.0, tolerance=0.0):
        self.long_ = long_
        self.short_ = short_
        self.hedge_ratio = abs(hedge_ratio)
        self.tolerance = abs(tolerance)
        self.constraint_assets = pd.Index([self.long_, self.short_])
        super(Pair, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s，%s，对冲比率：%s)" % (self.__class__.__name__, self.long_,
                                      self.short_, self.hedge_ratio)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        """
        参考原作者方法
        https://github.com/cvxgrp/cvxpy/issues/322
        """
        universe = init_weights.index
        long_ix, short_ix = universe.get_indexer([self.long_, self.short_])

        # 对应变量
        long_weight = weight_var[long_ix]
        short_weight = weight_var[short_ix]

        # The hedge ratio constraint is better understood logically as
        #       hedge_ratio = long_weight / -short_weight,
        #
        # to ensure that hedge_ratio is positive, but rearranged as
        #       hedge_ratio * -short_weight = long_weight,
        #
        # so that we cannot divide by 0. Then we add tolerance to make the
        # upper and lower bound constraints

        max_hedge_ratio = self.hedge_ratio + self.tolerance
        min_hedge_ratio = self.hedge_ratio - self.tolerance

        upper_bound_hedge_ratio_constraint = \
            (long_weight <= -short_weight * max_hedge_ratio)

        lower_bound_hedge_ratio_constraint = \
            (long_weight >= -short_weight * min_hedge_ratio)

        return [
            long_weight >= 0.0,
            short_weight <= 0.0,
            upper_bound_hedge_ratio_constraint,
            lower_bound_hedge_ratio_constraint,
        ]


class Basket(BaseConstraint):
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
        msg_fmt = '限制:{}, 最小值应小于等于最大值'
        assert min_net_exposure < max_net_exposure, msg_fmt.format(
            self.__class__.__name__)
        if not isinstance(assets, Iterable):
            assets = [assets]
        self.assets = assets
        self.min_net_exposure = min_net_exposure
        self.max_net_exposure = max_net_exposure
        self.constraint_assets = self.assets
        super(Basket, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(最小权重%s，最大权重%s)" % (self.__class__.__name__,
                                      self.min_net_exposure,
                                      self.max_net_exposure)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        locs = weight_var_series.index.get_indexer(self.assets)
        min_cons = [weight_var[loc] >= self.min_net_exposure for loc in locs]
        max_cons = [weight_var[loc] <= self.max_net_exposure for loc in locs]
        return min_cons + max_cons


class Frozen(BaseConstraint):
    """
    限定资产头寸不得更改

    Parameters
    ----------
    asset_or_assets : Asset or sequence[Asset]
        Asset(s) whose weight(s) cannot change.
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        # 如果输入单个资产，转换为可迭代对象
        if not isinstance(asset_or_assets, Iterable):
            asset_or_assets = [asset_or_assets]
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(self.asset_or_assets)
        super(Frozen, self).__init__()

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        assert init_weights is not None, 'Frozen()限制期初投资组合不得为空'
        constraints = []
        # 变量位置
        locs = weight_var_series.index.get_indexer(self.asset_or_assets)
        for loc, asset in zip(locs, self.asset_or_assets):
            constraints.append(weight_var[loc] == init_weights[asset])
        return constraints


class ReduceOnly(BaseConstraint):
    """
    限制资产的权重只能减少且不得低于零
    
    Parameters
    ----------
    asset：Asset
        不得增加头寸的资产
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        # 如果输入单个资产，转换为可迭代对象
        if not isinstance(asset_or_assets, Iterable):
            asset_or_assets = [asset_or_assets]
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(self.asset_or_assets)
        super(ReduceOnly, self).__init__()

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        assert init_weights is not None, 'ReduceOnly()期初投资组合不得为空'
        constraints = []
        locs = weight_var_series.index.get_indexer(self.asset_or_assets)
        for loc, asset in zip(locs, self.asset_or_assets):
            old_w = init_weights[asset]
            # 多头权重时，变量不得为空头
            if old_w >= 0:
                # 介于[0, old_w]
                constraints.extend(
                    [weight_var[loc] <= old_w, weight_var[loc] >= 0])
            else:
                # 介于[old_w, 0]
                constraints.extend(
                    [weight_var[loc] >= old_w, weight_var[loc] <= 0])
        return constraints


class LongOnly(BaseConstraint):
    """
    限定指定资产不得持有空头头寸

    Parameters
    ----------
    asset_or_assets：Asset or iterable[Asset]
        资产头寸必须为多头或为0
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        # 如果输入单个资产，转换为可迭代对象
        if not isinstance(asset_or_assets, Iterable):
            asset_or_assets = [asset_or_assets]
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(self.asset_or_assets)
        super(LongOnly, self).__init__()

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        constraints = []
        # 对特定资产的权重表达式限定为>=0
        locs = weight_var_series.index.get_indexer(self.asset_or_assets)
        for loc in locs:
            constraints.append(weight_var[loc] >= 0)
        return constraints


class ShortOnly(BaseConstraint):
    """
    限定指定资产不得持有多头头寸
    Parameters
    ----------
    asset_or_assets：Asset or iterable[Asset]
        资产头寸必须为空头或为0
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        # 如果输入单个资产，转换为可迭代对象
        if not isinstance(asset_or_assets, Iterable):
            asset_or_assets = [asset_or_assets]
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(asset_or_assets)
        super(ShortOnly, self).__init__()

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        constraints = []
        # 对特定资产的权重表达式限定为<=0
        locs = weight_var_series.index.get_indexer(self.asset_or_assets)
        for loc in locs:
            constraints.append(weight_var[loc] <= 0)
        return constraints


class FixedWeight(BaseConstraint):
    """
    限定单个资产的权重

    Parameters
    ----------
    asset：Asset
        资产
    weight：float
        权重值
    """

    def __init__(self, asset, weight):
        self.asset = asset
        self.weight = weight
        self.constraint_assets = asset
        super(FixedWeight, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(限定%s权重为%s)" % (self.__class__.__name__, self.asset,
                                  self.weight)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        expr = weight_var_series[self.asset]
        return [expr == self.weight]


class CannotHold(BaseConstraint):
    """
    限定资产头寸规模必须为0

    Parameters
    ----------
    asset_or_assets：Asset or iterable[Asset]
        不得持有头寸的资产或多个资产
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        # 如果输入单个资产，转换为可迭代对象
        if not isinstance(asset_or_assets, Iterable):
            asset_or_assets = [asset_or_assets]        
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(asset_or_assets)
        super(CannotHold, self).__init__()

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        constraints = []
        # 对特定资产的权重表达式限定为==0
        locs = weight_var_series.index.get_indexer(self.asset_or_assets)
        for loc in locs:
            constraints.append(weight_var[loc] == 0)
        return constraints


class NotExceed(BaseConstraint):
    """
    无论多头或空头，所有单个资产的权重不得超过一个常数
    权重介于 [-limit, limit]
    Parameters
    ----------
    limit：float 正数(负数将转换为绝对值)
        资产绝对值权重不得超过此值
    """

    def __init__(self, limit):
        limit = abs(limit)
        self.limit = limit
        super(NotExceed, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(区间[-%s,+%s])" % (self.__class__.__name__, self.limit,
                                        self.limit)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        constraints = []
        for i in range(len(weight_var_series)):
            constraints.extend([
                weight_var[i] <= self.limit,
                weight_var[i] >= -self.limit
            ])
        return constraints


class NotLessThan(BaseConstraint):
    """
    单个资产的权重不得低于一个常数

    Parameters
    ----------
    limit：float
        权重不得低于此值
    """

    def __init__(self, limit):
        self.limit = limit
        super(NotLessThan, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(权重>=%s)" % (self.__class__.__name__, self.limit)

    def to_cvxpy(self, weight_var, weight_var_series, init_weights):
        return [weight_var >= self.limit]
