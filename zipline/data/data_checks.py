#
# Copyright 2016 Quantopian, Inc.
#
from collections import namedtuple
import os

import numpy as np
from numpy import where

from zipline.pipeline.data.equity_pricing import USEquityPricing


Unpaired = namedtuple('Unpaired', ('dates', 'a', 'b'))


class UnpairedDailyBars(object):

    def __init__(self,
                 calendar,
                 asset_finder,
                 reader_a,
                 reader_b,
                 ignore_ipo=False,
                 assets=None):
        self.calendar = calendar
        self.asset_finder = asset_finder
        self.reader_a = reader_a
        self.reader_b = reader_b
        if assets is None:
            self.assets = sorted(asset_finder.retrieve_all(
                asset_finder.group_by_type(asset_finder.sids)['equity']))
        else:
            self.assets = assets

        self.assets = [asset for asset in self.assets if
                       (asset in self.reader_a._first_rows)
                       and
                       (asset in self.reader_b._first_rows)]
        self.field = USEquityPricing.volume

    def asset_path(self, asset):
        return os.path.join(
            self.rootdir, "{0}.json".format(int(asset)))

    def unpaired(self, start_date, end_date, ignore_ipo=False):
        result = {}
        data_a = self.reader_a.load_raw_arrays(
            [self.field], start_date, end_date, self.assets)[0]
        data_b = self.reader_b.load_raw_arrays(
            [self.field], start_date, end_date, self.assets)[0]
        start_loc = self.calendar.searchsorted(start_date)
        for i, asset in enumerate(self.assets):
            if asset.start_date >= start_date:
                if ignore_ipo:
                    asset_start_date = asset.start_date + self.calendar.freq
                else:
                    asset_start_date = asset.start_date
            else:
                asset_start_date = start_date
            asset_start_loc = self.calendar.searchsorted(asset_start_date)
            asset_end_loc = self.calendar.searchsorted(min(asset.end_date,
                                                           end_date))
            start = asset_start_loc - start_loc
            end = asset_end_loc - start_loc
            asset_data_a = data_a[start:end, i]
            asset_data_b = data_b[start:end, i]
            zero_a = asset_data_a == 0
            zero_b = asset_data_b == 0
            where_both = zero_a == zero_b
            if not np.all(where_both):
                where_diff = where(~where_both)[0]

                days_where_diff = self.calendar[where_diff + asset_start_loc]

                data = Unpaired(
                    days_where_diff,
                    asset_data_a[where_diff].tolist(),
                    asset_data_b[where_diff].tolist(),
                )
                result[asset] = data
        return result
