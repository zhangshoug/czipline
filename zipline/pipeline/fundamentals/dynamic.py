"""
动态信息
"""
import json
import os
import re
from functools import partial
from shutil import rmtree

import numpy as np
import pandas as pd
from logbook import Logger
from odo import odo

from cswd.common.utils import data_root, ensure_list
from cswd.sql.constants import MARGIN_MAPS

from .base import STOCK_DB, bcolz_table_path
from .utils import _normalize_ad_ts_sid

LABEL_MAPS = {'日期': 'asof_date', '股票代码': 'sid'}
DYNAMIC_TABLES = ['adjustments', 'margins', 'special_treatments','short_names']

logger = Logger('动态数据')


def normalized_dividend_data(df):
    """每股现金股利"""
    raw_df = df[['股票代码', '日期', '派息']].copy()
    raw_df.rename(columns=LABEL_MAPS, inplace=True)
    raw_df.rename(columns={'派息': 'amount'}, inplace=True)
    # 截止日期为发放日期前一天
    raw_df['asof_date'] = pd.to_datetime(raw_df['asof_date']) - pd.Timedelta(days=1)
    return raw_df


def normalized_margins_data(df):
    """融资融券数据"""
    df.drop(['更新时间', '序号'], axis=1, inplace=True)
    df.rename(columns=LABEL_MAPS, inplace=True)
    df.rename(columns={v: k for k, v in MARGIN_MAPS.items()}, inplace=True)
    return df

# # TODO:使用日线数据简化代替
def normalized_short_names_data(df):
    """股票简称数据"""
    # 股票未上市或者新上市，暂无股票代码
    df = df[~df['股票代码'].isna()].copy()
    df.drop(['更新时间', '序号', '备注说明'], axis=1, inplace=True)
    df.rename(columns=LABEL_MAPS, inplace=True)
    df.rename(columns={'股票简称': 'short_name'}, inplace=True)
    # 原为实施日期，截止日期前移一天
    df['asof_date'] = df['asof_date'] - pd.Timedelta(days=1)
    return df

# #　TODO:股票简称-> 转换
def normalized_special_treatments_data(df):
    """特殊处理历史"""
    # 股票未上市或者新上市，暂无股票代码
    df = df[~df['股票代码'].isna()]
    df = df[['股票代码', '日期', '特别处理']].copy()
    df.rename(columns=LABEL_MAPS, inplace=True)
    df.rename(columns={'特别处理': 'treatment'}, inplace=True)
    # 原为实施日期，截止日期前移一天
    df['asof_date'] = df['asof_date'] - pd.Timedelta(days=1) 
    return df


def _factory(table_name):
    if table_name == 'adjustments':
        return normalized_dividend_data
    elif table_name == 'short_names':
        return normalized_short_names_data
    elif table_name == 'special_treatments':
        return normalized_special_treatments_data
    elif table_name == 'margins':
        return normalized_margins_data
    raise NotImplementedError(table_name)


def _write_by_expr(expr, ndays=0):
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


def write_dynamic_data_to_bcolz(tables=None):
    """
    将每日变动数据以bcolz格式存储，提高数据集加载速度
    """
    if not tables:
        to_does = DYNAMIC_TABLES
    else:
        to_does = ensure_list(to_does)
    for table in to_does:
        if table in DYNAMIC_TABLES:
            ndays = 0
            logger.info('预处理表：{}'.format(table))
            expr = STOCK_DB[table]
            _write_by_expr(expr, ndays)
        else:
            raise ValueError('要写入的数据表{}不在"{}"范围'.format(
                table, DYNAMIC_TABLES))
