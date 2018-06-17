"""
基于基础数据的内置过滤器、分类器

*ST---公司经营连续三年亏损，退市预警。
ST----公司经营连续二年亏损，特别处理。
S*ST--公司经营连续三年亏损，退市预警+还没有完成股改。
SST---公司经营连续二年亏损，特别处理+还没有完成股改。
S----还没有完成股改
"""
import numpy as np
import pandas as pd

from ..utils.math_utils import nanmean, nansum
from ..utils.numpy_utils import changed_locations
from .data.equity_pricing import USEquityPricing
from .factors import CustomFactor, DailyReturns, SimpleMovingAverage
from .filters import CustomFilter, StaticSids
from .fundamentals.reader import Fundamentals

##################
# 辅助函数
##################


def continuous_num(bool_values, include=False):
    """尾部连续为真累计数"""
    assert bool_values.ndim == 2, "数据ndim必须为2"
    assert bool_values.dtype is np.dtype('bool'), '数据类型应为bool'

    def get_count(x):
        num = len(x)
        count = 0
        if not include:
            for i in range(num - 1, 0, -1):
                # 自尾部循环，一旦存在假，则返回
                if x[i] == 0:
                    break
            count += 1
        else:
            count = np.count_nonzero(x)
        return count

    # 本应使用已注释代码，但出现问题。原因不明？？？
    # return np.apply_along_axis(get_count, 0, bool_values)
    return pd.DataFrame(bool_values).apply(get_count).values


def quarterly_multiplier(dates):
    """
    季度乘数，用于预测全年数据（适用于累计数据的季度乘数）

    参数
    ----
        dates：报告日期（非公告日期）

    Notes
    -----
        1. 假设数据均匀分布
        2. 原始数据为累计数，2季度转换为年度则乘2,三季度乘以1.333得到全年

    例如：
        第三季度累计销售收入为6亿，则全年预测为8亿

        乘数 = 4 / 3
    """

    def func(x):
        return x.quarter

    if dates.ndim == 1:
        qs = pd.Series(dates).map(func)
    else:
        qs = pd.DataFrame(dates).applymap(func)
    return 4 / qs


def _select_annual_indices(dates):
    """选取最近第n年年末财务报告位置辅助函数"""
    col_num = dates.shape[1]
    locs = []
    for col in range(col_num):
        singal_dates = dates[:, col]
        # 规范时间
        singal_dates = pd.to_datetime(singal_dates).normalize()
        # 有可能存在NaT，以最小日期填充
        singal_dates = singal_dates.fillna(pd.Timestamp('1990-12-31'))
        loc = changed_locations(singal_dates, True)
        locs.append(loc)
    return locs


##################
# 自定义因子
##################


class SuccessiveYZ(CustomFactor):
    """
    尾部窗口一字数量

    Parameters
    ----------
    include : bool
        是否计算全部一字数量，若为否，则只计算尾部连续一字数量
        可选参数。默认为否。
    include_st : bool
        可选参数。是否包含ST股票。默认包含。
    window_length : int
        可选参数。默认为100天，不小于3。

    returns
    -------
        连续一字涨停数量， 连续一字跌停数量

    Notes:
    ------
        截至当前连续一字板数量。
        1. 最高 == 最低
        2. 涨跌幅超限
    """
    params = {'include_st': True, 'include': False}
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    outputs = ['zt', 'dt']
    window_length = 100

    def _validate(self):
        super(SuccessiveYZ, self)._validate()
        if self.window_length < 3:
            raise ValueError('window_length值必须大于2')

    def compute(self, today, assets, out, high, low, close, include_st,
                include):
        limit = 0.04 if include_st else 0.09
        # shape = (window_length - 1, N_assets)
        pct = (close[1:] - close[:-1]) / close[:-1]  # 当日涨跌幅，而非期间涨跌幅
        is_yz = (high == low)[1:]  # 保持形状一致
        is_zt = pct > limit
        is_dt = pct < -limit
        yz_zt = is_zt & is_yz
        yz_dt = is_dt & is_yz
        out.zt = continuous_num(yz_zt, include)
        out.dt = continuous_num(yz_dt, include)


class NDays(CustomFactor):
    """
    上市天数

    Notes
    -----
    当前日期与上市日期之间自然日历天数    
    """

    # 输入shape = (m days, n assets)
    inputs = [Fundamentals.info.asof_date]
    window_length = 1

    def compute(self, today, assets, out, ts):
        baseline = today.tz_localize(None)
        days = [(baseline - pd.Timestamp(x)).days for x in ts[0]]
        out[:] = days


class TradingDays(CustomFactor):
    """
    窗口期内所有有效交易天数

    Parameters
    ----------
    window_length : 整数
        统计窗口数量

    Notes
    -----
    如成交量大于0,表示当天交易
    """
    inputs = [USEquityPricing.volume]

    def compute(self, today, assets, out, vs):
        out[:] = np.count_nonzero(vs, 0)


class SuccessiveSuspensionDays(CustomFactor):
    """
    尾部窗口停牌交易日数

    Parameters
    ----------
    **Default Inputs:** None

    window_length：int
        窗口数量，可选，默认90天
        **注意**如果实际停牌天数超过`window_length`，最多`window_length`-1天
        设置window_length过大，有计算副作用
    include : bool
        是否计算窗口全部停牌，若为否，则只计算尾部连续停牌
        可选参数。默认为否。

    returns
    -------
        连续停牌天数
    """
    inputs = [USEquityPricing.amount]
    window_length = 90
    params = {'include': False}

    def compute(self, today, assets, out, vs, include):
        is_suspension = vs == 0
        out[:] = continuous_num(is_suspension, include)


##################
# 过滤器
##################


def IsST():
    """
    当前是否为ST状态

    Notes
    -----
    1. 使用基础数据中股票简称列最新状态值
    2. 如果简称中包含ST字符，则为ST
    """
    short_name = Fundamentals.short_name.latest
    return short_name.has_substring('ST')


def QTradableStocksUS():
    """
    可交易股票(过滤器)

    条件
    ----
        1. 该股票在过去200天内必须有180天的有效收盘价
        2. 并且在最近20天的每一天都正常交易(非停牌状态)
        以上均使用成交量来判定，成交量为0，代表当天停牌
    """
    v20 = TradingDays(window_length=20)
    v200 = TradingDays(window_length=200)
    return (v20 >= 20) & (v200 >= 180)


def IsNewShare(days=90):
    """
    次新股过滤器

    参数
    _____
    days：整数
        上市天数小于指定天数，判定为次新股
        默认90天

    returns
    -------
    zipline.pipeline.Filter
        次新股过滤器

    备注
    ----
        按自然日历计算上市天数
    """
    t_days = NDays()
    return t_days <= days


def TopAverageAmount(N=500, window_length=21):
    """
    成交额前N位过滤

    参数
    _____
    N：整数
        取前N位。默认前500。
    window_length：整数
        窗口长度。默认21个交易日。

    returns
    -------
    zipline.pipeline.Filter
        成交额前N位股票过滤器

    备注
    ----
        以窗口长度内平均成交额为标准        
    """
    high_amount = SimpleMovingAverage(
        inputs=[USEquityPricing.amount], window_length=window_length).top(N)
    return high_amount


class IsYZZT(CustomFilter):
    """
    是否一字涨停

    Parameters
    ----------
    **Default Inputs:** None

    **Default Window Length:** None

    include_st : bool
        是否包含st，默认包含st

    备注
    ----
    只要当天最高价等于最低价，且上涨，即为一字涨停。包含st涨停。
    """
    inputs = [USEquityPricing.close, USEquityPricing.high, USEquityPricing.low]
    window_safe = True
    window_length = 2
    params = {'include_st': True}

    def _validate(self):
        super(IsYZZT, self)._validate()
        if self.window_length != 2:
            raise ValueError('window_length值必须为2')

    def compute(self, today, assets, out, cs, hs, ls, include_st):
        is_yz = hs[-1] == ls[-1]
        if include_st:
            is_sz = cs[-1] > cs[0]
        else:
            # 涨幅超过6%，排除ST股票
            is_sz = (cs[-1] - cs[0]) / cs[0] > 0.06
        out[:] = is_yz & is_sz


class IsYZDT(CustomFilter):
    """
    是否一字跌停

    Parameters
    ----------
    **Default Inputs:** None

    **Default Window Length:** None

    include_st : bool
        是否包含st，默认包含st

    备注
    ----
    只要当天最高价等于最低价，且下跌，即为一字跌停。默认包含st跌停。
    """
    inputs = [USEquityPricing.close, USEquityPricing.high, USEquityPricing.low]
    window_safe = True
    window_length = 2
    params = {'include_st': True}

    def _validate(self):
        super(IsYZDT, self)._validate()
        if self.window_length != 2:
            raise ValueError('window_length值必须为2')

    def compute(self, today, assets, out, cs, hs, ls, include_st):
        is_yz = hs[-1] == ls[-1]
        if include_st:
            is_xd = cs[-1] < cs[0]
        else:
            # 涨幅超过6%，排除ST股票
            is_xd = (cs[-1] - cs[0]) / cs[0] < -0.06
        out[:] = is_yz & is_xd


class IsResumed(CustomFilter):
    """
    当日是否为停牌后复牌的首个交易日

    Parameters
    ----------
    **Default Inputs:** None

    **Default Window Length:** None

    备注
    ----
        首日成交量为0,次日有成交
        首日市值>0，排除新股上市
    """
    inputs = [USEquityPricing.amount, USEquityPricing.cmv]
    window_safe = True
    window_length = 2

    def _validate(self):
        super(IsResumed, self)._validate()
        if self.window_length != 2:
            raise ValueError('window_length值必须为2')

    def compute(self, today, assets, out, amount, cmv):
        is_tp = amount[0] == 0
        is_fp = (amount[-1] - amount[0]) > 0
        not_new = cmv[0] > 0  # 通过流通市值排除新股上市
        out[:] = is_tp & is_fp & not_new


def BlackNames(stock_codes):
    """
    股票黑名单过滤器

    ``StaticAssets`` is mostly useful for debugging or for interactively
    computing pipeline terms for a fixed set of assets that are known ahead of
    time.

    Parameters
    ----------
    stock_codes : iterable[stock code]
        An iterable of stock code for which to filter.

    用法
    ----
        如需要排除过滤掉黑名单，只需要在返回结果前添加`~`
    >>> black_stocks = ['000033','300156']
    >>> bns = BlackNames(black_stocks)
    >>> # 在pipeline参数`screen`中使用
    >>> screen = ~bns
    """
    assert isinstance(stock_codes, list), '必须以列表形式传入参数stock_codes'
    if len(stock_codes):
        sids = frozenset([int(stock_code) for stock_code in stock_codes])
    else:
        sids = frozenset([])
    return StaticSids(sids=sids) 


##################
# 财务相关因子
##################


class AnnualFinancalData(CustomFactor):
    """
    选取当前时间为t,t-n年的年度科目数据

    注意
    ----
        科目名称中必须含yearly，只对年报数据有效
    python
        inputs = [
            Fundamentals.profit_statement_yearly.A033,
            Fundamentals.profit_statement_yearly.report_end_date
        ]
    """
    window_length = 245
    params = {'t_n': 1}
    window_safe = True

    def _validate(self):
        super(AnnualFinancalData, self)._validate()
        # 验证输入列
        if len(self.inputs) != 2:
            raise ValueError('inputs列表长度只能为2。第一列为要查询的科目，第2列为report_end_date')
        last_col = self.inputs[-1]
        if last_col.name != 'report_end_date':
            raise ValueError('inputs列表最后一项必须为"report_end_date"')
        # 验证窗口长度与t_n匹配
        t_n = self.params.get('t_n')
        win = self.window_length
        at_least = t_n * 245
        if win < at_least:
            raise ValueError('window_length值至少应为t_n*245，即{}'.format(t_n * 245))

    def compute(self, today, assets, out, values, dates, t_n):
        locs = _select_annual_indices(dates)
        for col_loc, row_locs in enumerate(locs):
            if len(row_locs) < t_n:
                # 替代方案，防止中断
                row_loc = row_locs[0]
            else:
                row_loc = row_locs[-t_n]
            out[col_loc] = values[row_loc, col_loc]


class TTMSale(CustomFactor):
    """
    最近一年营业总收入季度平均数

    Notes:
    ------
    trailing 12-month (TTM)
    """
    inputs = [
        Fundamentals.profit_statement.A001,
        Fundamentals.profit_statement.asof_date
    ]
    # 一年的交易日不会超过250天
    window_length = 250

    def _validate(self):
        super(TTMSale, self)._validate()
        if self.window_length < 250:
            raise ValueError('window_length值必须或等于250,以确保获取一年的财务数据')

    def compute(self, today, assets, out, sales, asof_date):
        # 计算期间季度数发生变化的位置，简化计算量
        # 选取最近四个季度的日期所对应的位置
        loc = changed_locations(asof_date[:, 0], include_first=True)[-4:]
        adj = quarterly_multiplier(asof_date[loc])
        # 将各季度调整为年销售额，然后再平均
        res = nanmean(np.multiply(sales[loc], adj), axis=0)
        # 收入单位为万元
        out[:] = res * 10000


class TTMDividend(CustomFactor):
    """
    最近一年每股股利

    Notes:
    ------
    trailing 12-month (TTM)
    """
    inputs = [Fundamentals.dividend.amount, Fundamentals.dividend.asof_date]
    # 一年的交易日不会超过260天
    window_length = 260

    def _validate(self):
        super(TTMDividend, self)._validate()
        if self.window_length < 260:
            raise ValueError('window_length值必须大于或等于260,以确保获取一年的财务数据')

    def compute(self, today, assets, out, ds, asof_date):
        def func(x):
            return x.month

        mns = pd.Series(asof_date[:, 0]).map(func)
        # mns = np.apply_along_axis(func, 0, asof_date[:,0])
        # 选取最近12个月的日期所对应的位置
        loc = changed_locations(mns, include_first=True)[-12:]
        out[:] = nanmean(ds[loc], axis=0)


##################
# 估值相关因子
##################


# 参考网址：https://www.quantopian.com/help/fundamentals#valuation
def enterprise_value():
    """"""
    return NotImplementedError()


def market_cap():
    """总市值"""
    return USEquityPricing.tmv.latest


def market_cap_2():
    """流通市值"""
    return USEquityPricing.cmv.latest


def shares_outstanding():
    """总股本"""
    return USEquityPricing.total_share.latest


def shares_outstanding_2():
    """流通股本"""
    return USEquityPricing.circulating_share.latest


# 以下为估值相关比率


def book_value_per_share():
    """普通股股东权益/稀释流通股"""
    # 股东权益 / 实收资本
    return Fundamentals.balance_sheet.A107.latest / Fundamentals.balance_sheet.A095.latest


def book_value_yield():
    """BookValuePerShare / Price"""
    return book_value_per_share() / USEquityPricing.close.latest


def buy_back_yield():
    """The net repurchase of shares outstanding over the market capital of the company. 
    It is a measure of shareholder return."""
    return NotImplementedError()


def trailing_dividend_yield():
    """尾部12个月每股股利平均值/股价"""
    return TTMDividend() / USEquityPricing.close.latest


def earning_yield():
    """稀释每股收益率(稀释每股收益/价格)"""
    return Fundamentals.profit_statement.A045.latest / USEquityPricing.close.latest


def ev_to_ebitda():
    """
    This reflects the fair market value of a company, and allows comparability to other 
    companies as this is capital structure-neutral.
    """
    return NotImplementedError()


def fcf_per_share():
    """Free Cash Flow / Average Diluted Shares Outstanding"""
    return NotImplementedError()


# 别名
QTradableStocks = QTradableStocksUS
TAA = TopAverageAmount
