"""
使用Abstract API简化talib应用
"""
from talib import abstract


def indicators(func_name, ohlcv, *args, **kwargs):
    """
    获取单个股票期间talib函数指标

    参数
    ----
    func_name:str
        talib函数名称
    ohlcv:pd.DataFrame
        以时间为Index，ohlcv五列数据框
    args:func可接受的参数
    kwargs:func可接受的键值对参数
    返回
    ----
        如果指标为单列，返回pd.Series
        否则返回pd.DataFrame
    """
    func = abstract.Function(func_name, ohlcv, *args, **kwargs)
    return func.outputs