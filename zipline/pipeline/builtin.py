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

from .data.equity_pricing import USEquityPricing
from .factors import CustomFactor
from .fundamentals.reader import Fundamentals


#============================自定义因子=============================#
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


QTradableStocks = QTradableStocksUS


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
