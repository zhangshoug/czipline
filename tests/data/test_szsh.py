import unittest
from zipline.data.bundles.sqldata import fetch_single_equity, fetch_single_quity_adjustments
from zipline.data.bundles.szsh import _adjusted_raw_data
from zipline.data.us_equity_pricing import UINT32_MAX


class TestDailyData(unittest.TestCase):
    def setUp(self):
        self.symbol = symbol = '601857'
        self.start = start = '2018-1-1'
        self.end = end = '2018-4-18'
        self.daily_data = daily_data = fetch_single_equity(symbol, start, end)
        self.adjusted_daily_data = _adjusted_raw_data(daily_data)

    def test_data_overflow(self):
        origin_max_v = self.daily_data.tmv.max()
        self.assertGreater(origin_max_v, UINT32_MAX)
        adj_max_v = self.adjusted_daily_data.tmv.max()
        self.assertLess(adj_max_v, UINT32_MAX)
