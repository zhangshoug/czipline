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
from .filters import CustomFilter
from .fundamentals.reader import Fundamentals

#============================辅助函数区=============================#


def continuous_num(df):
    """计算连续数量"""
    msg = "数据框所有列类型都必须为`np.dtype('bool')`"
    assert all(df.dtypes == np.dtype('bool')), msg

    def compute(x):
        num = len(x)
        res = 0
        for i in range(num-1, 0, -1):
            if x[i] == 0:
                break
            else:
                res += 1
        return res
    return df.apply(compute, axis=0).values


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
    def func(x): return x.quarter
    if dates.ndim == 1:
        qs = pd.Series(dates).map(func)
    else:
        qs = pd.DataFrame(dates).applymap(func)
    return 4 / qs

#============================自定义因子=============================#


class SuccessiveYZZT(CustomFactor):
    """
    连续一字涨停数量

    Parameters
    ----------
    include_st : bool
        可选参数。是否包含ST股票。默认包含。

    returns
    -------
        连续一字涨停数量

    Notes:
    ------
        截至当前连续一字板数量。
        1. 最高 == 最低
        2. 涨停
    """
    params = {'include_st': True}
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]

    def compute(self, today, assets, out, high, low, close, include_st):
        limit = 0.04 if include_st else 0.09
        is_zt = pd.DataFrame(close).pct_change() >= limit
        is_yz = pd.DataFrame((high == low))
        yz_zt = is_zt & is_yz
        out[:] = continuous_num(yz_zt)


class SuccessiveYZDT(CustomFactor):
    """
    连续一字涨停数量

    Parameters
    ----------
    include_st : bool
        可选参数。是否包含ST股票。默认包含。

    returns
    -------
        连续一字跌停数量

    Notes:
    ------
        截至当前连续一字板数量。
        1. 最高 == 最低
        2. 跌停
    """
    params = {'include_st': True}
    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    # outputs = ['zt', 'dt']

    def compute(self, today, assets, out, high, low, close, include_st):
        limit = -0.04 if include_st else -0.09
        is_dt = pd.DataFrame(close).pct_change() < limit
        is_yz = pd.DataFrame((high == low))
        yz_zt = is_dt & is_yz
        out[:] = continuous_num(yz_zt)


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
    窗口期内有效交易天数

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


#================================过滤器=============================#


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
        1. 该股票在过去200天内必须有180天的有效收盘价(未实现)
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
    high_amount = SimpleMovingAverage(inputs=[USEquityPricing.amount],
                                      window_length=window_length).top(N)
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

#==============================财务相关=============================#


class TTMSale(CustomFactor):
    """
    最近一年营业总收入季度平均数

    Notes:
    ------
    trailing 12-month (TTM)
    """
    inputs = [Fundamentals.profit_statement.A001,
              Fundamentals.profit_statement.asof_date]
    # 一年的交易日不会超过260天
    window_length = 260

    def _validate(self):
        super(TTMSale, self)._validate()
        if self.window_length < 260:
            raise ValueError('window_length值必须大于260,以确保获取一年的财务数据')

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
    inputs = [Fundamentals.dividend.amount,
              Fundamentals.dividend.asof_date]
    # 一年的交易日不会超过260天
    window_length = 260

    def _validate(self):
        super(TTMDividend, self)._validate()
        if self.window_length < 260:
            raise ValueError('window_length值必须大于260,以确保获取一年的财务数据')

    def compute(self, today, assets, out, ds, asof_date):
        def func(x): return x.month
        mns = pd.Series(asof_date[:, 0]).map(func)
        # mns = np.apply_along_axis(func, 0, asof_date[:,0])
        # 选取最近12个月的日期所对应的位置
        loc = changed_locations(mns, include_first=True)[-12:]
        out[:] = nanmean(ds[loc], axis=0)


#==============================估值相关=============================#
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
