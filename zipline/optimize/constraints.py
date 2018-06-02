"""
限制模型
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from collections import Iterable
import cvxpy as cvx

from .utils import check_series_or_dict

__all__ = [
    'NotConstrained',
    'MaxGrossExposure',
    'NetExposure',
    'DollarNeutral',
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

    def get_ix(self, series, asset_or_assets):
        index = series.index
        ix = index.get_indexer(asset_or_assets)
        return ix

    def gen_constraints(self,
                        vars_long,
                        vars_short,
                        w_long_s=None,
                        w_short_s=None,
                        init_w_s=None):
        """
        返回限制列表

        Args
        ----
        vars_long：多头权重变量
        vars_short：空头权重变量
        w_long_s：多头权重表达式序列(以assets为索引)
        w_short：空头权重表达式序列(以assets为索引)
        init_w_s：初始权重值序列(以assets为索引)
        """
        return self._constraints_list(vars_long, vars_short, w_long_s,
                                      w_short_s, init_w_s)

    @abstractmethod
    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        # w = vars_long + cvx.abs(vars_short)
        # return [cvx.norm(w, 1) <= self.max_]
        return [cvx.sum(vars_long + cvx.abs(vars_short)) <= self.max_]


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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        total = cvx.sum(vars_long + vars_short)
        return [total >= self.min_, total <= self.max_]


class DollarNeutral(BaseConstraint):
    """
    限定多头与空头风险敞口必须等规模

    要求投资组合中多头权重累计与空头权重累计之差落在容差区间

    Parameters
    ----------
    tolerance : float, optional
        允许的净风险敞口偏移率。默认值是0.0001
    """

    def __init__(self, tolerance=0.0001):
        assert tolerance < 0.01, 'tolerance太大没有意义'
        self.tolerance = tolerance
        super(DollarNeutral, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s)" % (self.__class__.__name__, self.tolerance)

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        diff = cvx.sum(vars_long) + cvx.sum(vars_short)
        return [
            cvx.norm(diff, 1) <= self.tolerance,
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
    对于不应设置最低限制的组，请设置：

    min_weights[group_label] = opt.NotConstrained

    对于不应设置最高限制的组，请设置：

    max_weights[group_label] = opt.NotConstrained
    """

    def __init__(self, labels, min_weights, max_weights, etf_lookthru=None):
        msg = """
        每组都必须指定上下限。如果某组不需要上限或下限，请使用：
        min_weights[group_label] = opt.NotConstrained 或
        max_weights[group_label] = opt.NotConstrained
        """
        check_series_or_dict(labels, 'labels')
        check_series_or_dict(min_weights, 'min_weights')
        check_series_or_dict(max_weights, 'max_weights')
        labels = pd.Series(labels)
        min_weights = pd.Series(min_weights)
        max_weights = pd.Series(max_weights)
        assert min_weights.index.equals(max_weights.index), msg
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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        constraints = []

        def gen_constraints(expr_list, limit, is_gt):
            """内部生成限制列表"""
            if is_gt == '大于':
                constraints.extend([expr >= limit for expr in expr_list])
            else:
                constraints.extend([expr <= limit for expr in expr_list])

        labels = self.labels
        groups = self.max_weights.index  #.union(self.min_weights.index).unique()
        msg_fmt = '组"{}"最小值"{}"应小于等于最大值"{}"'
        for g in groups:
            min_ = self.min_weights[g]
            max_ = self.max_weights[g]
            assert min_ <= max_, msg_fmt.format(g, min_, max_)
            assets = labels[labels == g].index
            g_l_w = w_long_s.loc[assets].values  # 组多头表达式
            g_s_w = w_short_s.loc[assets].values  # 组空头表达式
            if min_ != 'NotConstrained':
                gen_constraints(g_l_w, min_, '大于')  # 多头大于最小值
                if min_ >= 0:
                    # 该组不允许空头
                    gen_constraints(g_s_w, 0, '大于')  # 空头设大于0
                else:
                    # 空头也大于最小值
                    gen_constraints(g_s_w, min_, '大于')
            if max_ != 'NotConstrained':
                gen_constraints(g_s_w, max_, '小于')
                # 不处理最大值限制
                if max_ <= 0:
                    # 该组不允许多头
                    gen_constraints(g_l_w, 0, '小于')
                else:
                    gen_constraints(g_l_w, max_, '小于')
        return constraints

    # def _constraints_list(self, w, vars_long, vars_short, assets):
    #     # 限定条件表达式列表
    #     constraints = []
    #     # 行业组
    #     groups = self.max_weights.index  #.union(self.min_weights.index)
    #     # 存放分组变量
    #     group_l_w = {g: [] for g in groups}
    #     group_s_w = {g: [] for g in groups}
    #     for i, idx in enumerate(assets):
    #         if idx in self.labels.index:
    #             g = self.labels[idx]
    #             group_l_w[g].append(vars_long[i])
    #             group_s_w[g].append(vars_short[i])
    #     msg_fmt = '分组{}中：最大值限制：{}应大于最小值限制：{}'
    #     for g in groups:
    #         min_limit = self.min_weights[g]
    #         max_limit = self.max_weights[g]
    #         assert max_limit > min_limit, msg_fmt.format(
    #             g, max_limit, min_limit)
    #         if min_limit < 0:
    #             expr_1 = cvx.sum(group_s_w[g]) >= min_limit
    #         else:
    #             expr_1 = cvx.sum(group_l_w[g]) >= min_limit
    #         if max_limit < 0:
    #             expr_2 = cvx.sum(group_s_w[g]) <= max_limit
    #         else:
    #             expr_2 = cvx.sum(group_l_w[g]) <= max_limit
    #         constraints.extend([expr_1, expr_2])
    #     return constraints

    # def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
    #                       init_w_s):
    #     # 限定条件表达式列表
    #     constraints = []
    #     # 行业组
    #     groups = self.max_weights.index.union(self.min_weights.index).unique()
    #     # 存放分组变量
    #     group_l_w = {g: [] for g in groups}
    #     group_s_w = {g: [] for g in groups}
    #     assets = w_long_s.index
    #     for i, idx in enumerate(assets):
    #         if idx in self.labels.index:
    #             g = self.labels[idx]
    #             group_l_w[g].append(vars_long[i])
    #             group_s_w[g].append(vars_short[i])
    #     # 添加下限
    #     for g in groups:
    #         min_limit = self.min_weights[g]
    #         if min_limit == NotConstrained:
    #             continue
    #         # 组多空变量列表
    #         g_v = group_s_w[g] + group_l_w[g]
    #         expr_g = cvx.sum(g_v)
    #         constraints.append(expr_g >= min_limit)
    #     # 添加上限
    #     for g in groups:
    #         max_limit = self.max_weights[g]
    #         if max_limit == NotConstrained:
    #             continue
    #         # 组变量列表
    #         g_v = group_s_w[g] + group_l_w[g]
    #         expr_g = cvx.sum(g_v)
    #         constraints.append(expr_g <= max_limit)
    #     return constraints


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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        # 限定条件表达式列表
        constraints = []
        d_min = self.default_min_weight
        d_max = self.default_max_weight
        assets = w_long_s.index
        candidates_min = self.min_weights[assets]
        candidates_max = self.max_weights[assets]
        candidates_min.fillna(d_min, inplace=True)
        candidates_max.fillna(d_max, inplace=True)
        assert all(candidates_max >= candidates_min), '单个股票的最大值应大于等于最小值'
        for i, (min_, max_) in enumerate(
                zip(candidates_min.values, candidates_max.values)):
            if min_ >= 0:
                # 最小值、最大值均位于零点右侧
                constraints.extend([
                    # 不允许空头
                    vars_short[i] >= 0,
                    vars_long[i] >= min_,
                    vars_long[i] <= max_
                ])
            else:
                if max_ >= 0:
                    # 最大值位于零点右侧，而最小值位于左侧
                    constraints.extend([
                        vars_long[i] >= 0, vars_long[i] <= max_,
                        vars_short[i] <= 0, vars_short[i] >= min_
                    ])
                else:
                    # 最小值、最大值均位于零点左侧
                    constraints.extend([
                        # 不允许多头
                        vars_long[i] <= 0,
                        vars_short[i] >= min_,
                        vars_short[i] <= max_
                    ])
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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        min_cons, max_cons = [], []
        # 按传入权重序列顺序重排，确保与变量顺序一致
        loadings = self.loadings.loc[w_long_s.index]
        w = vars_long + vars_short
        for c in loadings.columns:
            min_ = self.min_exposures[c]
            max_ = self.max_exposures[c]
            min_cons.append(cvx.sum(w * loadings[c]) >= min_)
            max_cons.append(cvx.sum(w * loadings[c]) <= max_)
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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        """
        参考原作者方法
        https://github.com/cvxgrp/cvxpy/issues/322
        """
        universe = init_w_s.index
        long_ix, short_ix = universe.get_indexer([self.long_, self.short_])

        # 对应变量
        long_weight = vars_long[long_ix]
        short_weight = vars_short[short_ix]

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
        assert min_net_exposure < max_net_exposure, '最小值应小于最大值'
        if not isinstance(assets, Iterable):
            assets = [assets]
        self.assets = assets
        self.min_net_exposure = min_net_exposure
        self.max_net_exposure = max_net_exposure
        self.constraint_assets = self.assets
        super(Basket, self).__init__()

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        min_cons, max_cons = [], []

        long_ix = w_long_s.index.get_indexer(self.assets)
        short_ix = w_short_s.index.get_indexer(self.assets)

        long_expr = vars_long[long_ix]
        short_expr = vars_short[short_ix]

        min_cons = [
            e_l + e_s >= self.min_net_exposure
            for e_l, e_s in zip(long_expr, short_expr)
        ]
        max_cons = [
            e_l + e_s <= self.max_net_exposure
            for e_l, e_s in zip(long_expr, short_expr)
        ]
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
        self.constraint_assets = self.asset_or_assets
        super(Frozen, self).__init__()

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        assert init_w_s is not None, '固定权重(Frozen)限制期初投资组合不得为空'
        long_cons, short_cons = [], []
        init_w = init_w_s[self.asset_or_assets]

        long_w = init_w[init_w >= 0]
        short_w = init_w[init_w < 0]

        if len(long_w):
            long_expr = w_long_s.loc[self.asset_or_assets].values
            long_cons = [e == b for e, b in zip(long_expr, long_w.values)]

        if len(short_w):
            short_expr = w_short_s.loc[self.asset_or_assets].values
            short_cons = [e == b for e, b in zip(short_expr, short_w.values)]

        return long_cons + short_cons


class ReduceOnly(BaseConstraint):
    """
    限制资产的权重只能减少且不得低于零
    
    Parameters
    ----------
    asset：Asset
        不得增加头寸的资产

    逻辑说明
    -------
    所有指定资产绝对值权重均不得增加，只能减少且不得小于0
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        # 如果输入单个资产，转换为可迭代对象
        if not isinstance(asset_or_assets, Iterable):
            asset_or_assets = [asset_or_assets]
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(self.asset_or_assets)
        super(ReduceOnly, self).__init__()

    # 分别控制多空权重
    # def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
    #                       init_w_s):
    #     long_cons, short_cons = [], []
    #     init_w = init_w_s.loc[self.asset_or_assets]
    #     init_long = init_w[init_w >= 0]
    #     init_short = init_w[init_w < 0]
    #     # 如果存在多头reduce限定
    #     if len(init_long):
    #         # 表达式减去值
    #         long_diff = w_long_s.loc[self.asset_or_assets] - init_long
    #         long_cons = [expr <= 0 for expr in long_diff]
    #     if len(init_short):
    #         short_expr = w_short_s.loc[self.asset_or_assets].values
    #         short_cons = [
    #             cvx.abs(e) <= cvx.abs(b)
    #             for e, b in zip(short_expr, init_short.values)
    #         ]
    #     return long_cons + short_cons

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        assert init_w_s is not None, '仅能减少权重限制(ReduceOnly)限制期初投资组合不得为空'
        constraints = []
        bias = 0.00001  # 由于求解器不接受严格不等式，使用偏差微调
        # 以绝对值来控制
        init_abs_w = init_w_s[self.asset_or_assets].abs().values
        if len(init_abs_w):
            long_expr = w_long_s[self.asset_or_assets].values
            short_expr = w_short_s[self.asset_or_assets].values
            constraints = [
                cvx.abs(e1) + cvx.abs(e2) <= b * (1 - bias)
                for e1, e2, b in zip(long_expr, short_expr, init_abs_w)
            ]
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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        # 对特定资产的空头权重表达式限定为大于0
        vars_w = w_short_s[self.asset_or_assets].tolist()
        return [expr >= 0 for expr in vars_w]


class ShortOnly(BaseConstraint):
    """
    限定指定资产不得持有多头头寸
    Parameters
    ----------
    asset_or_assets：Asset or iterable[Asset]
        资产头寸必须为空头或为0
    """

    def __init__(self, asset_or_assets, max_error_display=10):
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(asset_or_assets)
        super(ShortOnly, self).__init__()

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        # 对特定资产的多头权重表达式限定为小于0
        vars_w = w_long_s.loc[self.asset_or_assets].tolist()
        return [expr <= 0 for expr in vars_w]


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

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        if self.weight >= 0:
            expr = w_long_s[self.asset]
        else:
            expr = w_short_s[self.asset]
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
        self.asset_or_assets = asset_or_assets
        self.max_error_display = max_error_display
        self.constraint_assets = pd.Index(asset_or_assets)
        super(CannotHold, self).__init__()

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        # 把资产交集部分的表达式设置为0
        p1 = w_long_s.loc[self.asset_or_assets].tolist()
        p2 = w_short_s.loc[self.asset_or_assets].tolist()
        return [con == 0 for con in p1 + p2]


class NotExceed(BaseConstraint):
    """
    无论多头或空头，所有单个资产的权重不得超过一个常数
    权重介于 [-limit, limit]
    Parameters
    ----------
    limit：float 正数(负数将转换为绝对值)
        资产绝对值权重不得超过此值

    注
    --
        当求解最大alpha时，如存在收益率为0的alpha且权重未分配完毕，结果不正确?
        如随机分配权重，
    """

    def __init__(self, limit):
        limit = abs(limit)
        self.limit = limit
        super(NotExceed, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(所有权重区间[-%s,+%s])" % (self.__class__.__name__, self.limit,
                                        self.limit)

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        # 空头变量已经设定为非正
        # 多头变量已经设定为非负
        return [
            vars_long <= self.limit,
            cvx.abs(vars_short) <= self.limit,
        ]


class NotLessThan(BaseConstraint):
    """
    单个资产的总权重不得低于一个常数

    Parameters
    ----------
    limit：float 正数(负数将转换为绝对值)
        资产绝对值权重不得低于此值
    """

    def __init__(self, limit):
        limit = abs(limit)
        self.limit = limit
        super(NotLessThan, self).__init__()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(所有绝对值权重>=%s)" % (self.__class__.__name__, self.limit)

    def _constraints_list(self, vars_long, vars_short, w_long_s, w_short_s,
                          init_w_s):
        # return [vars_long >= self.limit, vars_short <= -self.limit]
        return [vars_long - vars_short >= self.limit]
