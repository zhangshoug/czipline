"""
用途：
    1. notebook研究
    2. 测试
"""
import os
from collections import Iterable

import pandas as pd

from cswd.common.utils import data_root, ensure_list, sanitize_dates
from zipline.utils.calendars import get_calendar
# from zipline import run_algorithm
from zipline.assets import Asset
from zipline.assets._assets import Equity
from zipline.data.bundles.core import load
from zipline.data.data_portal import DataPortal
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.fundamentals.reader import Fundamentals
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.loaders.blaze import global_loader

bundle = 'cndaily'  # 使用测试集时，更改为cntdaily。加快运行速度

bundle_data = load(bundle)

pipeline_loader = USEquityPricingLoader(bundle_data.equity_daily_bar_reader,
                                        bundle_data.adjustment_reader)

trading_calendar = get_calendar()

data_portal = DataPortal(
    bundle_data.asset_finder,
    trading_calendar,
    trading_calendar.first_session,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader)


def choose_loader(column):
    if column in USEquityPricing.columns:
        return pipeline_loader
    elif Fundamentals.has_column(column):
        return global_loader
    raise ValueError("`PipelineLoader`没有注册列 %s." % column)


def to_tdates(start, end):
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
    first_type = type(symbols_[0])
    for s in symbols_:
        type_ = type(s)
        if type_ != first_type:
            raise TypeError('symbols_列表不得存在混合类型')
    if first_type is Equity:
        return symbols_
    elif first_type is int:
        symbols_ = [str(s).zfill(6) for s in symbols_]

    if symbol_reference_date is not None:
        asof_date = pd.Timestamp(symbol_reference_date, tz='UTC')
    else:
        asof_date = pd.Timestamp('today', tz='UTC')
    res = finder.lookup_symbols(symbols_, asof_date)
    # 返回的是列表
    return res


def run_pipeline(pipe, start, end):
    dates, start_date, end_date = to_tdates(start, end)
    finder = bundle_data.asset_finder
    df = SimplePipelineEngine(
        choose_loader,
        dates,
        finder,
    ).run_pipeline(pipe, start_date, end_date)
    return df


# def _history(assets,
#              start,
#              end,
#              frequency,
#              field,
#              symbol_reference_date=None,
#              start_offset=0):
#     valid_fields = ('open', 'high', 'low', 'close', 'price', 'volume')
#     msg = '只接受单一字段，有效字段为{}'.format(valid_fields)
#     assert isinstance(field, str), msg
#     if frequency == 'daily':
#         frequency = '1d'
#         bundle = 'cndaily'
#     else:
#         frequency = '1m'
#         bundle = 'cnminutely'
#     dates, start_date, end_date = to_tdates(start, end)
#     if start_offset:
#         start_date -= start_offset * dates.freq
#     start_loc = dates.get_loc(start_date)
#     end_loc = dates.get_loc(end_date)
#     bar_count = end_loc - start_loc + 1
#     data_dir = data_root('webcache/history')
#     file_path = os.path.join(data_dir, '{}.pkl'.format(field))

#     def initialize(context):
#         context.stocks = symbols(assets, symbol_reference_date)

#     def handle_data(context, data):
#         history = data.history(context.stocks, field, bar_count, frequency)
#         history.to_pickle(file_path)

#     # 只需要在最后一个交易日来运行
#     run_algorithm(
#         end_date,
#         end_date,
#         initialize,
#         100000,
#         handle_data=handle_data,
#         bundle=bundle,
#         bm_symbol='000300')

#     # 判断传入asset数量
#     return pd.read_pickle(file_path)

# def prices(assets,
#            start,
#            end,
#            frequency='daily',
#            price_field='price',
#            symbol_reference_date=None,
#            start_offset=0):
#     """
#     获取指定股票期间收盘价(复权处理)
#     Parameters:

#         assets (int/str/Asset or iterable of same)
#             Identifiers for assets to load. Integers are interpreted as sids.
#             Strings are interpreted as symbols.
#         start (str or pd.Timestamp)
#             Start date of data to load.
#         end (str or pd.Timestamp)
#             End date of data to load.
#         frequency ({'minute', 'daily'}, optional)
#             Frequency at which to load data. Default is ‘daily’.
#         price_field ({'open', 'high', 'low', 'close', 'price'}, optional)
#             Price field to load. ‘price’ produces the same data as ‘close’,
#             but forward-fills over missing data. Default is ‘price’.
#         symbol_reference_date (pd.Timestamp, optional)
#             Date as of which to resolve strings as tickers. Default is the current day.
#         start_offset (int, optional)
#             Number of periods before start to fetch. Default is 0.
#             This is most often useful for calculating returns.

#     Returns:

#     prices (pd.Series or pd.DataFrame)
#         Pandas object containing prices for the requested asset(s) and dates.

#     Data is returned as a pd.Series if a single asset is passed.

#     Data is returned as a pd.DataFrame if multiple assets are passed.
#     """
#     valid_fields = ('open', 'high', 'low', 'close', 'price')
#     msg = '只接受单一字段，有效字段为{}'.format(valid_fields)
#     assert isinstance(price_field, str), msg
#     return _history(assets, start, end, frequency, price_field,
#                     symbol_reference_date, start_offset)


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
    msg = 'frequency只能选daily'
    assert frequency == 'daily', msg

    valid_fields = ('open', 'high', 'low', 'close', 'price', 'volume',
                    'amount', 'cmv', 'tmv', 'total_share', 'turnover')
    msg = '只接受单一字段，有效字段为{}'.format(valid_fields)
    assert isinstance(price_field, str), msg

    dates, start_date, end_date = to_tdates(start, end)

    if start_offset:
        start_date -= start_offset * dates.freq

    start_loc = dates.get_loc(start_date)
    end_loc = dates.get_loc(end_date)
    bar_count = end_loc - start_loc + 1

    assets = symbols(assets, symbol_reference_date=symbol_reference_date)

    return data_portal.get_history_window(assets, end_date, bar_count, '1d',
                                          price_field, frequency)


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


def volumes(assets,
            start,
            end,
            frequency='daily',
            symbol_reference_date=None,
            start_offset=0,
            use_amount=False):
    """
    获取资产期间成交量(或成交额)

    Parameters
    ----------
        assets (int/str/Asset or iterable of same)
            Identifiers for assets to load. Integers are interpreted as sids. 
            Strings are interpreted as symbols.
        start (str or pd.Timestamp)
            Start date of data to load.
        end (str or pd.Timestamp)
            End date of data to load.
        frequency ({'minute', 'daily'}, optional)
            Frequency at which to load data. Default is ‘daily’.
        symbol_reference_date (pd.Timestamp, optional)
            Date as of which to resolve strings as tickers. Default is the current day.
        start_offset (int, optional)
            Number of periods before start to fetch. Default is 0. 
            This is most often useful for calculating returns.
        use_amount：bool
            是否使用成交额字段。默认为否。
            如使用成交额，则读取期间成交额数据。

    Returns:	

    volumes (pd.Series or pd.DataFrame)
        Pandas object containing volumes for the requested asset(s) and dates.

    Data is returned as a pd.Series if a single asset is passed.

    Data is returned as a pd.DataFrame if multiple assets are passed.
    """
    field = 'amount' if use_amount else 'volume'
    return prices(assets, start, end, frequency, field, symbol_reference_date,
                  start_offset)


def ohlcv(asset,
          start=pd.datetime.today() - pd.Timedelta(days=365),
          end=pd.datetime.today()):
    """获取单个股票期间ohlcv五列数据框"""
    fields = ['open', 'high', 'low', 'close', 'volume']
    dfs = []
    # 取单个股票价格数据，如果输入为Equity实例，转换为列表形式
    if isinstance(asset, Equity):
        asset = [asset]
    for field in fields:
        # 取单个股票价格数据，必须以列表形式输入[asset]
        df = prices(asset, start, end, price_field=field)
        dfs.append(df)
    res = pd.concat(dfs, axis=1)
    res.columns = fields
    return res.dropna()  # 在当天交易时，如果数据尚未刷新，当天会含有na
