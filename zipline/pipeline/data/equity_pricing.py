"""
Dataset representing OHLCV data.
"""
from zipline.utils.numpy_utils import float64_dtype

from .dataset import Column, DataSet


# class USEquityPricing(DataSet):
#     """
#     Dataset representing daily trading prices and volumes.
#     """
#     open = Column(float64_dtype)
#     high = Column(float64_dtype)
#     low = Column(float64_dtype)
#     close = Column(float64_dtype)
#     volume = Column(float64_dtype)


class USEquityPricing(DataSet):
    """
    Dataset representing daily trading prices and volumes.
    """
    open = Column(float64_dtype)
    high = Column(float64_dtype)
    low = Column(float64_dtype)
    close = Column(float64_dtype)
    volume = Column(float64_dtype)
    prev_close = Column(float64_dtype)
    turnover = Column(float64_dtype)
    amount = Column(float64_dtype)
    tmv = Column(float64_dtype)
    cmv = Column(float64_dtype)
    circulating_share = Column(float64_dtype) # # 流通股本
    total_share = Column(float64_dtype)       # # 总股本
    
# 别名
CNEquityPricing = USEquityPricing