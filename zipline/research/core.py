"""
用途：
    1. notebook研究
    2. 测试
"""
import os
import pandas as pd
from collections import Iterable

from cswd.common.utils import sanitize_dates, data_root, ensure_list
from zipline import run_algorithm
from zipline.assets import Asset
from zipline.pipeline import Pipeline
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.fundamentals import Fundamentals
from zipline.data.bundles.core import load
from zipline.pipeline.loaders.blaze import global_loader

bundle = 'cndaily'  # 使用测试集时，更改为cntdaily。加快运行速度

bundle_data = load(bundle)

pipeline_loader = USEquityPricingLoader(bundle_data.equity_daily_bar_reader,
                                        bundle_data.adjustment_reader)


def choose_loader(column):
    if column in USEquityPricing.columns:
        return pipeline_loader
    elif Fundamentals.has_column(column):
        return global_loader
    raise ValueError("`PipelineLoader`没有注册列 %s." % column)


def _dates(start, end):
    """修正交易日期"""
    calendar = bundle_data.equity_daily_bar_reader.trading_calendar
    dates = calendar.all_sessions
    # 修正日期
    start, end = sanitize_dates(start, end)
    # 定位交易日期
    start_date = dates[dates.get_loc(start, method='bfill')]
    end_date = dates[dates.get_loc(end, method='ffill')]
    if start_date > end_date:
        start_date = end_date
    return dates, start_date, end_date


def symbols(symbols_, symbol_reference_date=None, handle_missing='log'):
    """
    Convert a string or a list of strings into Asset objects.
    Parameters:	

        symbols_ (String or iterable of strings.)
            Passed strings are interpreted as ticker symbols and 
            resolved relative to the date specified by symbol_reference_date.
        symbol_reference_date (str or pd.Timestamp, optional)
            String or Timestamp representing a date used to resolve symbols 
            that have been held by multiple companies. Defaults to the current time.
        handle_missing ({'raise', 'log', 'ignore'}, optional)
            String specifying how to handle unmatched securities. Defaults to ‘log’.

    Returns:	

    list of Asset objects – The symbols that were requested.
    """
    finder = bundle_data.asset_finder
    symbols_ = ensure_list(symbols_)
    symbols_ = [str(s).zfill(6) if isinstance(s, int) else s for s in symbols_]
    if symbol_reference_date is not None:
        asof_date = pd.Timestamp(symbol_reference_date, tz='UTC')
    else:
        asof_date = pd.Timestamp('today', tz='UTC')
    res = finder.lookup_symbols(symbols_, asof_date)
    if len(res) == 1:
        return res[0]
    else:
        return res


def run_pipeline(pipe, start, end):
    dates, start_date, end_date = _dates(start, end)
    finder = bundle_data.asset_finder
    df = SimplePipelineEngine(
        choose_loader,
        dates,
        finder,
    ).run_pipeline(pipe, start_date, end_date)
    return df


def prices(assets,
           start,
           end,
           frequency='daily',
           price_field='price',
           symbol_reference_date=None,
           start_offset=0):
    """
    获取指定股票期间收盘价(复权处理)
    Parameters:	

        assets (int/str/Asset or iterable of same)
            Identifiers for assets to load. Integers are interpreted as sids. 
            Strings are interpreted as symbols.
        start (str or pd.Timestamp)
            Start date of data to load.
        end (str or pd.Timestamp)
            End date of data to load.
        frequency ({'minute', 'daily'}, optional)
            Frequency at which to load data. Default is ‘daily’.
        price_field ({'open', 'high', 'low', 'close', 'price'}, optional)
            Price field to load. ‘price’ produces the same data as ‘close’, 
            but forward-fills over missing data. Default is ‘price’.
        symbol_reference_date (pd.Timestamp, optional)
            Date as of which to resolve strings as tickers. Default is the current day.
        start_offset (int, optional)
            Number of periods before start to fetch. Default is 0. 
            This is most often useful for calculating returns.

    Returns:	

    prices (pd.Series or pd.DataFrame)
        Pandas object containing prices for the requested asset(s) and dates.

    Data is returned as a pd.Series if a single asset is passed.

    Data is returned as a pd.DataFrame if multiple assets are passed.   
    """
    valid_fields = ('open', 'high', 'low', 'close', 'price')
    msg = '只接受单一字段，有效字段为{}'.format(valid_fields)
    assert isinstance(price_field, str), msg
    if frequency == 'daily':
        frequency = '1d'
        bundle = 'cndaily'
    else:
        frequency = '1m'
        bundle = 'cnminutely'
    dates, start_date, end_date = _dates(start, end)
    if start_offset:
        start_date -= start_offset * dates.freq
    start_loc = dates.get_loc(start_date)
    end_loc = dates.get_loc(end_date)
    bar_count = end_loc - start_loc + 1
    data_dir = data_root('webcache/history')
    file_path = os.path.join(data_dir, 'history.pkl')

    def initialize(context):
        context.stocks = symbols(assets, symbol_reference_date)

    def handle_data(context, data):
        history = data.history(context.stocks, price_field, bar_count,
                               frequency)
        history.to_pickle(file_path)

    # 只需要在最后一个交易日来运行
    run_algorithm(
        end_date,
        end_date,
        initialize,
        100000,
        handle_data=handle_data,
        bundle=bundle,
        bm_symbol='000300')

    # 判断传入asset数量
    return pd.read_pickle(file_path)


def returns(assets,
            start,
            end,
            periods=1,
            frequency='daily',
            price_field='price',
            symbol_reference_date=None):
    """
    Fetch returns for one or more assets in a date range.
    Parameters:	

        assets (int/str/Asset or iterable of same)
            Identifiers for assets to load.
            Integers are interpreted as sids.
            Strings are interpreted as symbols.
        start (str or pd.Timestamp)
            Start date of data to load.
        end (str or pd.Timestamp)
            End date of data to load.
        periods (int, optional)
            Number of periods over which to calculate returns. 
            Default is 1.
        frequency ({'minute', 'daily'}, optional)
            Frequency at which to load data. Default is ‘daily’.
        price_field ({'open', 'high', 'low', 'close', 'price'}, optional)
            Price field to load. ‘price’ produces the same data as ‘close’, 
            but forward-fills over missing data. Default is ‘price’.
        symbol_reference_date (pd.Timestamp, optional)
            Date as of which to resolve strings as tickers. 
            Default is the current day.

    Returns:	

    returns (pd.Series or pd.DataFrame)
        Pandas object containing returns for the requested asset(s) and dates.

    Data is returned as a pd.Series if a single asset is passed.

    Data is returned as a pd.DataFrame if multiple assets are passed.
    """
    df = prices(assets, start, end, frequency, price_field,
                symbol_reference_date, periods)
    return df.pct_change(periods).dropna()