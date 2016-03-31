#
# Copyright 2016 Quantopian, Inc.
#

import numpy as np
from numpy import isclose, where
from pandas import DataFrame

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


def compare(calendar, reader_a, reader_b, assets, start_date, end_date):
    results = {}
    start_loc = calendar.get_loc(start_date)

    for field in OHLCV:
        data_a = reader_a.load_raw_arrays(
            OHLCV, start_date, end_date, assets)[0]
        data_b = reader_b.load_raw_arrays(
            OHLCV, start_date, end_date, assets)[0]
        for i, asset in enumerate(assets):
            asset_start_loc = calendar.get_loc(
                max(asset.start_date, start_date))
            asset_end_loc = calendar.get_loc(min(asset.end_date, end_date))
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
