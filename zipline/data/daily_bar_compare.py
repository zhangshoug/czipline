#
# Copyright 2016 Quantopian, Inc.
#
import glob
import json
import os

import numpy as np
from numpy import isclose, where
from pandas import DataFrame, Timestamp, to_datetime

from zipline.pipeline.data.equity_pricing import USEquityPricing

OHLCV = (
    USEquityPricing.open,
    USEquityPricing.high,
    USEquityPricing.low,
    USEquityPricing.close,
    USEquityPricing.volume,
)


class PricingComp(object):

    def __init__(self, asset, diff_frame):
        self.asset = asset
        self.diff_frame = diff_frame

    def __repr__(self):
        return "PricingComp: asset={0} diff_frame={1}".format(
            self.asset,
            self.diff_frame.head() if self.diff_frame is not None else None)


class _Processed(object):

    def __init__(self, rootdir):
        self._processed_path = os.path.join(rootdir, 'processed.txt')
        self._processed_append_fp = None

    def processed(self):
        with open(self.processed_path, mode='r') as f:
            self._processed = [int(x) for x in f.readlines()]
        return self._processed

    def add_processed(self, sid):
        self._processed.append(sid)
        if self._processed_append_fp is None:
            self._processed_append_fp = open(self._processed_path, mode='ab')
        self._processed_append_fp.write(sid)
        self._processed_append_fp.flush()


class DailyBarComparison(object):

    def __init__(self,
                 rootdir,
                 calendar,
                 asset_finder,
                 reader_a,
                 reader_b,
                 start_date,
                 end_date,
                 assets=None,
                 fields=None):
        self.comp_reader = DailyBarComparisonReader(rootdir, asset_finder)
        self.calendar = calendar
        self.output_dir = rootdir
        self.reader_a = reader_a
        self.reader_b = reader_b
        self.start_date = start_date
        self.end_date = end_date
        if assets is None:
            self.assets = sorted(asset_finder.retrieve_all(asset_finder.sids))
        else:
            self.assets = assets
        if fields is None:
            self.fields = OHLCV
        else:
            self.fields = fields

        self.metadata = _Processed(rootdir)

    def compare(self):
        data_a = self.reader_a.load_raw_arrays(
            self.fields, self.start_date, self.end_date, self.assets)
        data_b = self.reader_b.load_raw_arrays(
            self.fields, self.start_date, self.end_date, self.assets)
        start_loc = self.calendar.get_loc(self.start_date)
        for i, asset in enumerate(self.assets):
            asset_start_loc = self.calendar.get_loc(
                max(asset.start_date, self.start_date))
            asset_end_loc = self.calendar.get_loc(min(asset.end_date,
                                                      self.end_date))
            start = asset_start_loc - start_loc
            end = asset_end_loc - start_loc
            asset_data_a = data_a[start:end, i]
            asset_data_b = data_b[start:end, i]
            isclose_arr = isclose(asset_data_a, asset_data_b,
                                  equal_nan=True)
            if np.all(isclose_arr):
                results[asset] = PricingComp(asset, None)
                continue

            where_diff = where(~isclose_arr)[0]

            days_where_diff = calendar[where_diff + asset_start_loc]

            diff_frame = DataFrame(index=days_where_diff, data={
                'a': asset_data_a[where_diff],
                'b': asset_data_b[where_diff]}
            )

            results[asset] = PricingComp(asset, diff_frame)
        return results


class DailyBarComparisonReader(object):

    def __init__(self, output_dir, asset_finder):
        self.output_dir = output_dir

    def assets(self):
        for asset_file in glob.glob(os.path.join(self.output_dir, "*.csv")):
            pass
