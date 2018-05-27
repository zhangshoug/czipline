"""
A股交易日历数据辅助函数

注：
    需要使用cswd包
    数据来源于本地数据库
"""
import pandas as pd
from cswd.sql.models import TradingCalendar
from cswd.sql.base import session_scope


def get_adhoc_holidays():
    """从本地数据库中获取特别假期(务必以UTC时区表达)"""
    today = pd.Timestamp('today')
    with session_scope() as sess:
        query = sess.query(
            TradingCalendar.date
        ).filter(
            TradingCalendar.date <= today.date()
        ).filter(
            TradingCalendar.is_trading == False
        )
        start = today + pd.Timedelta(days=1)
        end = today + pd.Timedelta(days=365)
        # 所有未来日期周六周日均视同为假期
        future = pd.date_range(start, end, tz='UTC')
        wds = pd.date_range(start, end, tz='UTC', freq='B')
        future_holidays = future.difference(wds)
        holidays = [x[0] for x in query.all()]
        holidays = pd.DatetimeIndex(holidays, tz='UTC')
        return holidays.union(future_holidays)
