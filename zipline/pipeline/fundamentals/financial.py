"""
定期财务报告

自巨潮提取上市公司预约与实际公布财务报告日期后，对已经有明确的公告日期的，时间戳更改为公告日期，
截止日期为公告日期前一天。对无法获取公告日期的，以报告日期后45天作为时间戳，前偏移一天为截止日期。

timestamp = 公告日期 如果公告日期存在
timestamp = 报告截止日期 + 45天 如果不存在公告日期

asof_date = timestamp - 1天

不再使用utils._normalize_ad_ts_sid, 改在本地完成
"""
import os
from functools import partial
from shutil import rmtree

import numpy as np
import pandas as pd
from logbook import Logger
from odo import odo
from cswd.tasks.report_dates import BookReportDate, flush_report_dates
from cswd.common.utils import data_root, ensure_list
from cswd.sql.constants import (BALANCESHEET_ITEM_MAPS,
                                CASHFLOWSTATEMENT_ITEM_MAPS, CHNL_ITEM_MAPS,
                                CZNL_ITEM_MAPS, PROFITSTATEMENT_ITEM_MAPS,
                                YLNL_ITEM_MAPS, YYNL_ITEM_MAPS, ZYZB_ITEM_MAPS)

from .base import STOCK_DB, bcolz_table_path
from .constants import QUARTERLY_TABLES
logger = Logger('财务报告')


def get_preprocessed_dates():
    """预处理实际披露时间表"""
    col_names = ['report_end_date', 'timestamp', 'code']
    brd = BookReportDate()
    dates = brd.data
    # 不需要预约及变更日期，加快写入及读取速度
    dates = dates[['报告期', '实际披露', '股票代码']]
    dates.columns = col_names
    for col in col_names:
        # 转换日期类型(原始日期值为str类型) -> Timestamp
        if col != 'code':
            dates[col] = dates[col].map(lambda x: pd.Timestamp(x))
    return dates


def normalize_ad_ts_sid(df, preprocessed_dates):
    """完成对实际公告日期的修正，转换股票代码为sid"""
    # 合并前，将report_end_date -> Timestamp
    df['report_end_date'] = df['report_end_date'].map(
        lambda x: pd.Timestamp(x))
    out = df.merge(
        preprocessed_dates, 
        how='left', on=['code', 'report_end_date'],
        validate='one_to_one',
    )
    # 存在公告日期
    has_adate = ~out['timestamp'].isna()
    # 先赋值截止日期为报告日期的后45天
    out['asof_date'] = out['report_end_date'] + pd.Timedelta(days=45)
    # 修正截止日期：对于存在公告日期的，将其设定为公告日期前一天，以此作为截止日期
    out.loc[
        has_adate,
        'asof_date'] = out.loc[has_adate, 'timestamp'] - pd.Timedelta(days=1)
    # 修正timestamp
    out.loc[~has_adate, 'timestamp'] = out.loc[~has_adate, 'asof_date'] + pd.Timedelta(days=1)
    # 改变类型
    out['sid'] = out['code'].map(lambda x: int(x))
    out['asof_date'] = pd.to_datetime(out['asof_date'])
    out['timestamp'] = pd.to_datetime(out['timestamp'])
    out['report_end_date'] = pd.to_datetime(out['report_end_date'])
    # 舍弃code列，保留report_end_date列
    out.drop(['code'], axis=1, inplace=True)
    out.sort_values(['sid', 'report_end_date'], inplace=True)
    return out


def normalized_financial_data(df, preprocessed_dates, maps):
    """财务报告数据"""
    df.drop(['更新时间', '序号', '公告日期'], axis=1, inplace=True)
    df.rename(
        columns={
            '股票代码': 'code',
            '报告日期': 'report_end_date'
        }, inplace=True)
    df.rename(columns={v: k for k, v in maps.items()}, inplace=True)
    return normalize_ad_ts_sid(df, preprocessed_dates)


def _factory(table_name):
    if table_name == 'balance_sheets':
        return partial(normalized_financial_data, maps=BALANCESHEET_ITEM_MAPS)
    elif table_name == 'profit_statements':
        return partial(
            normalized_financial_data, maps=PROFITSTATEMENT_ITEM_MAPS)
    elif table_name == 'cashflow_statements':
        return partial(
            normalized_financial_data, maps=CASHFLOWSTATEMENT_ITEM_MAPS)
    elif table_name == 'zyzbs':
        return partial(normalized_financial_data, maps=ZYZB_ITEM_MAPS)
    elif table_name == 'ylnls':
        return partial(normalized_financial_data, maps=YLNL_ITEM_MAPS)
    elif table_name == 'chnls':
        return partial(normalized_financial_data, maps=CHNL_ITEM_MAPS)
    elif table_name == 'cznls':
        return partial(normalized_financial_data, maps=CZNL_ITEM_MAPS)
    elif table_name == 'yynls':
        return partial(normalized_financial_data, maps=YYNL_ITEM_MAPS)
    raise NotImplementedError(table_name)


def _write_by_expr(expr):
    """
    以bcolz格式写入表达式数据

    转换步骤：
        1. 读取表数据转换为pd.DataFrame

    """
    table_name = expr._name

    # 整理数据表

    # 1. 读取表
    df = odo(expr, pd.DataFrame)

    # 2. 获取预先处理的上市公司公告日期表
    preprocessed_dates = get_preprocessed_dates()

    # 3. 调整转换数据
    processed = _factory(table_name)(df, preprocessed_dates)

    # 4. 转换为bcolz格式并存储
    rootdir = bcolz_table_path(table_name)
    if os.path.exists(rootdir):
        rmtree(rootdir)
    odo(processed, rootdir)
    logger.info('表:{}，数据存储路径：{}'.format(table_name, rootdir))


def write_financial_data_to_bcolz(tables=None):
    """
    将财务报告相关表以bcolz格式存储，提高数据集加载速度

    Notes:
    ------
        写入财务报告数据时，默认财务报告公告日期为报告期后45天
    """
    # 首先要刷新数据
    flush_report_dates()
    if not tables:
        to_does = QUARTERLY_TABLES
    else:
        to_does = ensure_list(to_does)
    for table in to_does:
        if table in QUARTERLY_TABLES:
            logger.info('预处理表：{}'.format(table))
            expr = STOCK_DB[table]
            _write_by_expr(expr)
        else:
            raise ValueError('要写入的数据表{}不在"{}"范围'.format(
                table, QUARTERLY_TABLES))
