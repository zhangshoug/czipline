"""
定期财务报告

自巨潮提取上市公司预约与实际公布财务报告日期后，对已经有明确的公告日期的，时间戳更改为公告日期，
截止日期为公告日期前一天。对无法获取公告日期的，以报告日期后45天作为时间戳，前偏移一天为截止日期。

timestamp = 公告日期 如果公告日期存在
timestamp = 报告截止日期 + 45天 如果不存在公告日期

asof_date = timestamp - 1天
"""
import os
from functools import partial
from shutil import rmtree

import numpy as np
import pandas as pd
from logbook import Logger
from odo import odo

from cswd.common.utils import data_root, ensure_list
from cswd.sql.constants import (BALANCESHEET_ITEM_MAPS,
                                CASHFLOWSTATEMENT_ITEM_MAPS, CHNL_ITEM_MAPS,
                                CZNL_ITEM_MAPS, PROFITSTATEMENT_ITEM_MAPS,
                                YLNL_ITEM_MAPS, YYNL_ITEM_MAPS, ZYZB_ITEM_MAPS)

from .base import STOCK_DB, bcolz_table_path
from .utils import _normalize_ad_ts_sid

QUARTERLY_TABLES = ['balance_sheets', 'profit_statements', 'cashflow_statements',
                    'chnls', 'cznls', 'ylnls', 'yynls', 'zyzbs']

logger = Logger('财务报告')


def normalized_financial_data(df, maps):
    """财务报告数据"""
    df.drop(['更新时间', '序号', '公告日期'], axis=1, inplace=True)
    df.rename(columns={'股票代码': 'code', '报告日期': 'date'}, inplace=True)
    df.rename(columns={v: k for k, v in maps.items()}, inplace=True)
    return df.fillna(value=0.0)


def _factory(table_name):
    if table_name == 'balance_sheets':
        return partial(normalized_financial_data,
                       maps=BALANCESHEET_ITEM_MAPS)
    elif table_name == 'profit_statements':
        return partial(normalized_financial_data,
                       maps=PROFITSTATEMENT_ITEM_MAPS)
    elif table_name == 'cashflow_statements':
        return partial(normalized_financial_data,
                       maps=CASHFLOWSTATEMENT_ITEM_MAPS)
    elif table_name == 'zyzbs':
        return partial(normalized_financial_data,
                       maps=ZYZB_ITEM_MAPS)
    elif table_name == 'ylnls':
        return partial(normalized_financial_data,
                       maps=YLNL_ITEM_MAPS)
    elif table_name == 'chnls':
        return partial(normalized_financial_data,
                       maps=CHNL_ITEM_MAPS)
    elif table_name == 'cznls':
        return partial(normalized_financial_data,
                       maps=CZNL_ITEM_MAPS)
    elif table_name == 'yynls':
        return partial(normalized_financial_data,
                       maps=YYNL_ITEM_MAPS)
    raise NotImplementedError(table_name)


def _write_by_expr(expr, ndays=45):
    """
    以bcolz格式写入表达式数据

    转换步骤：
        1. 读取表数据转换为pd.DataFrame

    """
    table_name = expr._name

    # 整理数据表

    # 1. 读取表
    df = odo(expr, pd.DataFrame)

    # 2. 调整转换数据
    processed = _factory(table_name)(df)

    # 3. 列名称及类型标准化
    out = _normalize_ad_ts_sid(processed, ndays)

    # 转换为bcolz格式并存储
    rootdir = bcolz_table_path(table_name)
    if os.path.exists(rootdir):
        rmtree(rootdir)
    odo(out, rootdir)
    logger.info('表:{}，数据存储路径：{}'.format(table_name, rootdir))


def write_financal_data_to_bcolz(tables=None):
    """
    将财务报告相关表以bcolz格式存储，提高数据集加载速度

    Notes:
    ------
        写入财务报告数据时，默认财务报告公告日期为报告期后45天
    """
    if not tables:
        to_does = QUARTERLY_TABLES
    else:
        to_does = ensure_list(to_does)
    for table in to_does:
        if table in QUARTERLY_TABLES:
            ndays = 45
            logger.info('预处理表：{}'.format(table))
            expr = STOCK_DB[table]
            _write_by_expr(expr, ndays)
        else:
            raise ValueError('要写入的数据表{}不在"{}"范围'.format(
                table, QUARTERLY_TABLES))
