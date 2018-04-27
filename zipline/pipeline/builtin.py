"""
基于基础数据的内置过滤器、分类器
"""
import numpy as np

from .data.equity_pricing import USEquityPricing
from .factors import CustomFactor
from .fundamentals.reader import Fundamentals


#============================自定义因子=============================#

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
