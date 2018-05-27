from unittest import TestCase
import pandas as pd

from .test_trading_calendar import ExchangeCalendarTestBase
from zipline.utils.calendars.exchange_calendar_szsh import SZSHExchangeCalendar


class SZSHCalendarTestCase(ExchangeCalendarTestBase, TestCase):
    answer_key_filename = "szsh"
    calendar_class = SZSHExchangeCalendar

    MAX_SESSION_HOURS = 5.5

    def test_2016_holidays(self):
        # 元旦：   1月1日
        # 春节：   2月7日~2月13日
        # 清明节： 4月4日
        # 劳动节:  5月1日~5月2日
        # 端午节:  6月9日~6月11日
        # 中秋节： 9月15日~9月17日
        # 国庆节： 10月1日~10月7日
        # 次年元旦：
        for day in ["2016-01-01", "2016-02-07", "2016-02-08", "2016-02-09",
                    "2016-02-10", "2016-02-11", "2016-02-12", "2016-02-13",
                    "2016-05-01", "2016-05-02", "2016-06-09", "2016-06-10",
                    "2016-06-11", "2016-09-15", "2016-09-16", "2016-09-17",
                    "2016-10-01", "2016-10-02", "2016-10-03", "2016-10-04",
                    "2016-10-05", "2016-10-06", "2016-10-07", "2017-01-01"]:
            self.assertFalse(
                self.calendar.is_session(pd.Timestamp(day, tz='UTC'))
            )

    def test_2016_early_closes(self):
        # 提前收盘2016-1-4,2016-1-7
        dt_1 = pd.Timestamp("2016-01-04", tz='UTC')
        self.assertTrue(dt_1 in self.calendar.early_closes)

        dt_2 = pd.Timestamp("2016-01-07", tz='UTC')
        self.assertTrue(dt_2 in self.calendar.early_closes)

        market_close = self.calendar.schedule.loc[dt_1].market_close
        market_close = market_close.tz_localize("UTC").tz_convert(
            self.calendar.tz
        )
        self.assertEqual(13, market_close.hour)
        self.assertEqual(35, market_close.minute)

    def test_adhoc_holidays(self):
        # 选择周末测试
        # 选择清明测试
        # 10月3日
        for day in ["1994-10-03", "2017-04-03", "2017-04-04",
                    "2017-10-04", "2018-04-14", "2018-04-15"]:
            self.assertFalse(
                self.calendar.is_session(pd.Timestamp(day, tz='UTC'))
            )
    