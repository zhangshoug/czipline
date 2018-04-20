"""
构造股票日线数据集
TODO：检查日期是否类型：datetime64[s]？datetime64[ns]？
"""

import pandas as pd
from logbook import Logger

from zipline.utils.calendars import register_calendar_alias

from . import core as bundles
from ..constants import ADJUST_FACTOR, TEST_SYMBOLS
from .sqldata import (fetch_single_equity, fetch_single_quity_adjustments,
                      gen_asset_metadata)

log = Logger(__name__)

# sid 没有实际意义。按符号查询即可。
# def _to_sid(x):
#     """符号转换为sid"""
#     return int(x)


# def _to_symbol(x):
#     """sid转换为符号"""
#     return str(x).zfill(6)


def _adjusted_raw_data(raw_df):
    """调整原始数据单位，转换适用于unit32类型的数值"""
    data = raw_df.copy()
    # 增强兼容性
    for col in data.columns:
        if col in ADJUST_FACTOR.keys():
            data.loc[:, col] = data[col] * ADJUST_FACTOR.get(col, 1)
    return data


def _update_splits(splits, asset_id, origin_data):
    if origin_data.empty:
        # 如为空表，直接返回，不进行任何处理
        return
    # 调整适应于zipline算法
    # date -> datetime64[ns]
    df = pd.DataFrame({'ratio': 1 / (1 + origin_data.ratio),
                       'effective_date': pd.to_datetime(origin_data.listing_date),
                       'sid': asset_id})
    splits.append(df)


def _update_dividends(dividends, asset_id, origin_data):
    if origin_data.empty:
        return
    # 原始数据中存在 'record_date' 'pay_date'列，但没有'declared_date'列
    origin_data['declared_date'] = pd.NaT
    # date -> datetime64[ns]
    origin_data['record_date'] = pd.to_datetime(origin_data['record_date'])
    origin_data['pay_date'] = pd.to_datetime(origin_data['pay_date'])
    origin_data['ex_date'] = origin_data['pay_date']
    origin_data['sid'] = asset_id
    dividends.append(origin_data)


def gen_symbol_data(symbol_map,
                    sessions,
                    splits,
                    dividends):
    for asset_id, symbol in symbol_map.iteritems():
        # asset_id = _to_sid(symbol)
        # 日线原始数据
        raw_data = fetch_single_equity(
            symbol,
            start=sessions[0],
            end=sessions[-1],
        )
        # 不得包含涨跌幅列，含有正负号
        raw_data.drop('change_pct', axis=1, inplace=True)
        # 调整数据精度
        raw_data = _adjusted_raw_data(raw_data)
        # 以日期、符号为索引
        raw_data.set_index(['date', 'symbol'], inplace=True)
        # 时区调整，以0.0填充na
        # 转换为以日期为索引的表(与sessions保持一致)
        asset_data = raw_data.xs(
            symbol,
            level=1
        ).reindex(
            sessions.tz_localize(None)
        ).fillna(0.0)
        # 顺带处理分红派息
        # 获取原始调整数据
        raw_adjustment = fetch_single_quity_adjustments(symbol,
                                                        start=sessions[0],
                                                        end=sessions[-1])
        # 当非空时才执行
        if not raw_adjustment.empty:
            # 送转有效日期为红股上市日期，大于0才有意义
            raw_splits = raw_adjustment.loc[raw_adjustment.ratio > 0.0,
                                            ['listing_date', 'ratio']]
            _update_splits(splits, asset_id, raw_splits)
            # 只需要登记日、支付日、金额列
            raw_dividends = raw_adjustment.loc[raw_adjustment.amount > 0.0,
                                               ['record_date', 'pay_date', 'amount']]
            _update_dividends(dividends, asset_id, raw_dividends)

        yield asset_id, asset_data

# 注意，注册函数名称要区别于quandl模块，否则会出错。名称随意。
@bundles.register('cnstock', calendar_name='SZSH', minutes_per_day=241)
def cnquandl_bundle(environ,
                    asset_db_writer,
                    minute_bar_writer,
                    daily_bar_writer,
                    adjustment_writer,
                    calendar,
                    start_session,
                    end_session,
                    cache,
                    show_progress,
                    output_dir):
    """Build a zipline data bundle from the cnstock dataset.
    """
    log.info('读取股票元数据......')
    metadata = gen_asset_metadata()

    symbol_map = metadata.symbol
    sessions = calendar.sessions_in_range(start_session, end_session)

    log.info('日线数据集（股票数量：{}）'.format(len(symbol_map)))

    # 写入股票元数据
    if show_progress:
        log.info('写入资产元数据')
    asset_db_writer.write(metadata)

    splits = []
    dividends = []
    daily_bar_writer.write(
        gen_symbol_data(
            symbol_map,
            sessions,
            splits,
            dividends,
        ),
        show_progress=show_progress,
    )

    adjustment_writer.write(
        splits=pd.concat(splits, ignore_index=True),
        dividends=pd.concat(dividends, ignore_index=True),
    )

# 以cn开头，方可匹配相应的读写器


@bundles.register('cntest', calendar_name='SZSH', minutes_per_day=241)
def cntest_bundle(environ,
                  asset_db_writer,
                  minute_bar_writer,
                  daily_bar_writer,
                  adjustment_writer,
                  calendar,
                  start_session,
                  end_session,
                  cache,
                  show_progress,
                  output_dir):
    """Build a zipline test data bundle from the cnstock dataset.
    """
    log.info('读取股票元数据......')
    # 测试集
    metadata = gen_asset_metadata()
    metadata = metadata[metadata.symbol.isin(TEST_SYMBOLS)]
    sessions = calendar.sessions_in_range(start_session, end_session)

    symbol_map = metadata.symbol
    log.info('生成测试数据集（共{}只）'.format(len(symbol_map)))
    splits = []
    dividends = []

    asset_db_writer.write(metadata)

    daily_bar_writer.write(
        gen_symbol_data(
            symbol_map,
            sessions,
            splits,
            dividends,
        ),
        show_progress=show_progress,
    )

    adjustment_writer.write(
        splits=pd.concat(splits, ignore_index=True),
        dividends=pd.concat(dividends, ignore_index=True),
    )


register_calendar_alias("CNSTOCK", "SZSH")
