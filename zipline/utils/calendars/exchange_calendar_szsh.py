from datetime import time
import pandas as pd
from pytz import timezone

from zipline.utils.calendars import TradingCalendar

from .szsh_data_utils import get_adhoc_holidays


class SZSHExchangeCalendar(TradingCalendar):
    """
    A股(深圳证券交易所、上海证券交易所)交易日历

    开盘时间: 09:30am, Asia/Shanghai
    收盘时间: 15:00pm, Asia/Shanghai
    午休开始: 11:31am
    午休结束: 12:59am

    处理说明：
    1. 对假期日历简化处理，自然日历 - 交易日历 = 非交易日，非交易日全部列为特别假期
    2. 所有分钟，在TradingCalendar原算法基础上，排除午休期间的分钟。所以正常情况下，
       每天分钟总数量为241个
    3. 未考虑中途暂停交易(不会影响回测，此时交易量为0)
    """
    # 默认为True
    use_lunch_break = True

    @property
    def name(self):
        return "SZSH"

    @property
    def tz(self):
        return timezone('Asia/Shanghai')

    @property
    def open_time(self):
        return time(9, 31)

    @property
    def close_time(self):
        return time(15)

    @property
    def lunch_break_start_time(self):
        return time(11, 31)

    @property
    def lunch_break_end_time(self):
        return time(12, 59)

    @property
    def special_closes_adhoc(self):
        """
        提前收盘时间(历史熔断)

        注：
        ---
        仅仅包含提前收盘，不含中途暂停交易(冷却期)

        Returns
        -------
        list: List of (time, DatetimeIndex) tuples that represent special
         closes that cannot be codified into rules.        
        """
        return [(time(13, 35), ['2016-01-04']),
                (time(10, 00), ['2016-01-07']), ]

    @property
    def adhoc_holidays(self):
        return get_adhoc_holidays()

    @property
    def actual_last_session(self):
        """最后交易日"""
        now = pd.Timestamp.utcnow()
        trading_days = self.all_sessions
        actual = trading_days[trading_days.get_loc(now, method='ffill')]
        return actual
