"""
Module for building a complete daily dataset from Quandl's WIKI dataset.
"""
from numbers import Number
from time import time, sleep
from urllib import urlencode

from bcolz import ctable
from logbook import Logger
import pandas as pd
from six import iteritems

from zipline.utils.control_flow import invert_unique
from zipline.utils.tradingcalendar import trading_days
from zipline.utils.input_validation import expect_types

from . import paths as pth
from .us_equity_pricing import BcolzDailyBarWriter, check_uint32_safe, OHLC

log = Logger('Quandl Loader')

QUANDL_TO_ZIPLINE = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}

ZIPLINE_TO_QUANDL = invert_unique(QUANDL_TO_ZIPLINE)


def quandl_data_path(paths, environ=None):
    return pth.data_path(['quandl'] + paths, environ=environ)


def ensure_quandl_root(environ=None):
    return pth.ensure_directory(quandl_data_path([]))


def raw_csv_path(symbol, environ=None):
    return quandl_data_path(['raw', symbol + '.csv'], environ=environ)


def asset_db_path(environ=None):
    return quandl_data_path(['assets.sqlite'], environ=environ)


def daily_equity_path(environ=None):
    return quandl_data_path(['equity_daily.bcolz'], environ=environ)


def ctable_from_dict(d, **bcolz_kwargs):
    """
    Construct a ctable from a dict mapping column name to numpy array.
    """
    names, columns = zip(*iteritems(d))
    return ctable(columns=columns, names=names, **bcolz_kwargs)


def fetch_symbols_frame():
    """
    Download Quandl symbol definitions.
    """
    data = pd.read_csv(
        "https://s3.amazonaws.com/"
        "quandl-static-content/Ticker+CSV%27s/WIKI_tickers.csv",
    ).rename(
        columns={
            "quandl code": "symbol",
            "name": "asset_name",
        }
    )
    data.symbol = data.symbol.str.replace("WIKI/", "")
    return data.sort('symbol')


def format_wiki_url(api_key, symbol, start_date, end_date):
    """
    Build a query URL for a quandl WIKI dataset.
    """
    query_params = [
        ('start_date', start_date.strftime('%Y-%m-%d')),
        ('end_date', end_date.strftime('%Y-%m-%d')),
        ('order', 'asc'),
    ]
    if api_key is not None:
        query_params = [('api_key', api_key)] + query_params

    return (
        "https://www.quandl.com/api/v3/datasets/WIKI/"
        "{symbol}.csv?{query}".format(
            symbol=symbol,
            query=urlencode(query_params),
        )
    )


def fetch_single_equity(api_key, symbol, start_date, end_date, retries=5):
    """
    Download data for a single equity.
    """
    for i in range(retries):
        try:
            return pd.read_csv(
                format_wiki_url(api_key, symbol, start_date, end_date),
                parse_dates=['Date'],
                index_col='Date',
                usecols=[
                    'Open',
                    'High',
                    'Low',
                    'Close',
                    'Volume',
                    'Date',
                    'Ex-Dividend',
                    'Split Ratio',
                ],
                na_values=['NA'],
            )
        except:
            log.exception("Exception raised reading Quandl data. Retrying.")
    else:
        raise Exception(
            "Failed to download data for %r after %d attempts." % (
                symbol, retries
            )
        )


def append_splits(equity_frame, sid, out):
    """
    Append split entries from `equity_frame` to ``splits_list``.

    Parameters
    ----------
    equity_frame : pd.DataFrame
        A DataFrame containing the following columns:

    """
    pass


def append_dividends(equity_frame, sid, out):
    """
    Append dividend entries from `equity_frame` to ``splits_list``.

    Parameters
    ----------
    equity_frame : pd.DataFrame
        A DataFrame containing the following columns:
        Date : datetime64[ns]
            The ex-date of a dividend.
        Ex-Dividend : float
            The dollar amount payed for the dividend.
    """
    pass


class QuandlDailyBarWriter(BcolzDailyBarWriter):
    """
    A daily bar writer that consumes from the Quandl WIKI dataset.

    After a call to `write` has been made, the writer also provides `splits`
    and `dividends` attributes containing dataframes of split and dividend
    information suitable for passing to a SQLiteAdjustmentWriter.
    """

    @expect_types(api_key=str, seconds_per_call=Number)
    def __init__(self, api_key, seconds_per_call, symbol_map):
        self._api_key = api_key
        self._seconds_per_call = seconds_per_call
        self._symbol_map = symbol_map

        self._splits = []
        self._dividends = []

    def gen_tables(self, assets, calendar):
        start_date = calendar[0]
        end_date = calendar[-1]
        for asset_id in assets:
            symbol = self._symbol_map[asset_id]
            start_time = time()
            raw_data = fetch_single_equity(
                self._api_key,
                symbol,
                start_date=start_date,
                end_date=end_date,
            )
            self._update_splits(raw_data)
            self._update_dividends(raw_data)

            raw_data = raw_data.reindex(calendar, copy=False).fillna(0.0)

            yield asset_id, self._format_raw_data(raw_data)

            self._sleep_for_rate_limit(time() - start_time)

    @property
    def progress_bar_message(self):
        return "Downloading from Quandl:"

    def progress_bar_item_show_func(self, sid_and_table):
        if sid_and_table is None:
            return "Initializing"
        return self._symbol_map[sid_and_table[0]]

    @staticmethod
    def to_uint32(array, colname):
        return array

    def _format_raw_data(self, raw_data):
        """
        Convert raw DataFrame from Quandl into a ctable for superclass API.
        """
        columns = {}
        for colname in OHLC:
            coldata = raw_data[ZIPLINE_TO_QUANDL[colname]]
            check_uint32_safe(coldata.max() * 1000, colname)
            columns[colname] = (coldata * 1000).astype('uint32')

        dates = raw_data.index.values.astype('datetime64[s]')
        check_uint32_safe(dates.max().view(int), 'day')
        columns['day'] = dates.astype('uint32')

        volumes = raw_data['Volume']
        check_uint32_safe(volumes.max(), 'volume')

        columns['volume'] = volumes.astype('uint32')
        return ctable_from_dict(columns)

    def _sleep_for_rate_limit(self, last_api_call_time):
        remaining = self._seconds_per_call - last_api_call_time
        if remaining > 0:
            sleep(remaining)

    def _update_splits(self, raw_data):
        pass

    def _update_dividends(self, raw_data):
        pass


def build_database(api_key,
                   symbol_map,
                   calendar,
                   show_progress=True,
                   environ=None):
    """
    Build a Zipline Database from Quandl data.
    """
    ensure_quandl_root(environ=environ)

    # Quandl rate-limits their API to 2000 API calls per 10 minutes.  We have
    # to make sure we don't exceeed that limit.
    seconds_per_call = (pd.Timedelta('10 minutes') / 2000).total_seconds()
    writer = QuandlDailyBarWriter(
        api_key,
        seconds_per_call=seconds_per_call,
        symbol_map=symbol_map,
    )

    pricing_root = quandl_data_path(
        ['daily_equities.bcolz'],
        environ=environ,
    )
    writer.write(
        filename=pricing_root,
        calendar=calendar,
        assets=list(symbol_map.index),
        show_progress=show_progress,
    )


if __name__ == '__main__':
    # calendar = trading_days[trading_days < pd.Timestamp('now', tz='UTC')]
    # symbol_frame = fetch_symbols_frame()
    # symbol_map = symbol_frame.symbol

    # build_database(
    #     'MY API KEY',
    #     symbol_map,
    #     calendar,
    #     show_progress=True,
    # )
