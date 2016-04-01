#
# Copyright 2016 Quantopian, Inc.
#
import click

from collections import namedtuple
import simplejson
import os

import numpy as np
from numpy import where

from zipline.pipeline.data.equity_pricing import USEquityPricing


Unpaired = namedtuple('Unpaired', ('dates', 'a', 'b'))


class _Processed(object):

    def __init__(self, rootdir):
        self._processed_path = os.path.join(rootdir, 'processed.txt')
        self._processed_append_fp = None
        self._has_diff_path = os.path.join(rootdir, 'with_diff.txt')
        self._has_diff_append_fp = None

    def processed(self):
        with open(self._processed_path, mode='r') as f:
            self._processed = [int(x) for x in f.readlines()]
        return self._processed

    def has_diff(self):
        with open(self._has_diff_path, mode='r') as f:
            self._has_diff = [int(x) for x in f.readlines()]
        return self._has_diff

    def add_processed(self, sid):
        if self._processed_append_fp is None:
            self._processed_append_fp = open(self._processed_path, mode='ab')
        self._processed_append_fp.write(str(sid))
        self._processed_append_fp.write("\n")
        self._processed_append_fp.flush()

    def add_has_diff(self, sid):
        if self._has_diff_append_fp is None:
            self._has_diff_append_fp = open(self._has_diff_path, mode='ab')
        self._has_diff_append_fp.write(str(sid))
        self._has_diff_append_fp.write("\n")
        self._has_diff_append_fp.flush()


class DailyBarComparison(object):

    def __init__(self,
                 rootdir,
                 calendar,
                 asset_finder,
                 reader_a,
                 reader_b,
                 start_date,
                 end_date,
                 assets=None):
        self.rootdir = rootdir
        self.calendar = calendar
        self.asset_finder = asset_finder
        self.reader_a = reader_a
        self.reader_b = reader_b
        self.start_date = start_date
        self.end_date = end_date
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
        self._processed = _Processed(rootdir)

    def asset_path(self, asset):
        return os.path.join(
            self.rootdir, "{0}.json".format(int(asset)))

    def compare(self):
        data_a = self.reader_a.load_raw_arrays(
            [self.field], self.start_date, self.end_date, self.assets)[0]
        data_b = self.reader_b.load_raw_arrays(
            [self.field], self.start_date, self.end_date, self.assets)[0]
        start_loc = self.calendar.searchsorted(self.start_date)
        for i, asset in enumerate(self.assets):
            asset_start_loc = self.calendar.searchsorted(
                max(asset.start_date, self.start_date))
            asset_end_loc = self.calendar.searchsorted(min(asset.end_date,
                                                           self.end_date))
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

                path = self.asset_path(asset)

                dates = [str(x).split()[0] for x in days_where_diff]
                data = Unpaired(
                    dates,
                    asset_data_a[where_diff].tolist(),
                    asset_data_b[where_diff].tolist(),
                )

                with open(path, mode='w') as fp:
                    simplejson.dump(data, fp)
                self._processed.add_has_diff(int(asset))
            self._processed.add_processed(int(asset))

    def unpaired_values(self):
        result = {}
        for sid in self._processed.has_diff():
            path = self.asset_path(sid)
            with open(path, mode='r') as fp:
                data = simplejson.load(fp)
                result[sid] = Unpaired(**data)
        return result


@click.command()
@click.option(
    '--a',
    type=click.Path(),
)
@click.option(
    '--b',
    type=click.Path(),
)
@click.option(
    '--output',
    type=click.Path(),
)
@click.option(
    '--assets-db-path',
    type=click.Path(),
)
@click.option(
    '--min-date',
    type=click.STRING,
)
@click.option(
    '--max-date',
    type=click.STRING,
)
def find_unpaired_bars(a, b, output, assets_db_path, min_date, max_date):
    from zipline.data.us_equity_pricing import BcolzDailyBarReader
    from zipline.finance.trading import TradingEnvironment
    from pandas import Timestamp
    reader_a = BcolzDailyBarReader(a)
    reader_b = BcolzDailyBarReader(b)

    if not os.path.exists(output):
        os.makedirs(output)

    min_date = Timestamp(min_date, tz='UTC')
    max_date = Timestamp(max_date, tz='UTC')

    env = TradingEnvironment(min_date=min_date,
                             max_date=max_date,
                             asset_db_path=assets_db_path)

    daily_bar_comparison = DailyBarComparison(
        output,
        env.trading_days,
        env.asset_finder,
        reader_a,
        reader_b,
        min_date,
        max_date,
    )
    daily_bar_comparison.compare()
    results = daily_bar_comparison.unpaired_values()
    import nose; nose.tools.set_trace()
    assert True

if __name__ == '__main__':
    find_unpaired_bars()
