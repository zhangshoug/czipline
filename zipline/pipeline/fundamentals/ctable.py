"""
使用bcolz存储数据

设定定期计划任务，自动完成sql数据到bcolz的格式转换
使用bcolz默认参数压缩，初步测试结果表明，大型数据加载至少提速一倍

"""
import os
import json
from shutil import rmtree
import pandas as pd
import numpy as np
from functools import partial
from logbook import Logger
from odo import odo
import re

from cswd.common.utils import data_root, ensure_list
from cswd.sql.constants import (
    BALANCESHEET_ITEM_MAPS,
    PROFITSTATEMENT_ITEM_MAPS,
    CASHFLOWSTATEMENT_ITEM_MAPS,
    MARGIN_MAPS,
    ZYZB_ITEM_MAPS,
    YLNL_ITEM_MAPS,
    CHNL_ITEM_MAPS,
    CZNL_ITEM_MAPS,
    YYNL_ITEM_MAPS,
)
from cswd.sql.base import session_scope
from cswd.sql.models import Category, StockCategory, Issue
from .base import STOCK_DB
from .utils import _normalize_ad_ts_sid

logger = Logger('数据转换')

QUARTERLY_TABLES = ['balance_sheets', 'profit_statements', 'cashflow_statements',
                    'chnls', 'cznls', 'ylnls', 'yynls', 'zyzbs']

INCLUDE_TABLES = ['special_treatments','adjustments', 'margins', 'issues'] + QUARTERLY_TABLES
PS_TABLES = ['regions', 'thx_industries', 'csrc_industries']
HANDLE_TABLES = ['concepts']
LABEL_MAPS = {'日期': 'asof_date', '股票代码': 'sid'}
CONCEPT_CODE_PATTERN = re.compile(r'\d{8}')


def _bcolz_table_path(table_name):
    """bcolz文件路径"""
    root_dir = data_root('bcolz')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    path_ = os.path.join(root_dir, '{}.bcolz'.format(table_name))
    return path_


def normalized_dividend_data(df):
    """处理现金分红（调整到年初）"""
    raw_df = df[['股票代码', '日期', '派息']].copy()
    raw_df.rename(columns=LABEL_MAPS, inplace=True)
    raw_df.rename(columns={'派息': 'amount'}, inplace=True)
    raw_df.index = pd.to_datetime(raw_df['asof_date'])
    res = raw_df.groupby('sid').resample('AS').sum().fillna(0.0)
    return res.reset_index()


def normalized_margins_data(df):
    """融资融券数据"""
    df.drop(['更新时间', '序号'], axis=1, inplace=True)
    df.rename(columns=LABEL_MAPS, inplace=True)
    df.rename(columns={v: k for k, v in MARGIN_MAPS.items()}, inplace=True)
    return df


def normalized_short_names_data(df):
    """股票简称数据"""
    df.drop(['更新时间', '序号', '备注说明'], axis=1, inplace=True)
    df.rename(columns=LABEL_MAPS, inplace=True)
    df.rename(columns={'股票简称': 'short_name'}, inplace=True)
    return df

def normalized_special_treatments_data(df):
    """股票简称数据"""
    df = df[['股票代码','日期','特别处理']].copy()
    df.rename(columns=LABEL_MAPS, inplace=True)
    df.rename(columns={'特别处理': 'treatment'}, inplace=True)
    return df

def normalized_ipo_data(df):
    """上市日期数据"""
    df = df[['上市日期', '股票代码']].copy()
    df.rename(columns={'股票代码': 'code', '上市日期': 'date'}, inplace=True)
    df['ipo'] = df['date']
    return df


def normalized_region_data():
    """股票所处地区数据"""
    with session_scope() as sess:
        query = sess.query(
            StockCategory.stock_code,
            Issue.A004_上市日期,
            Category.title,
        ).filter(
            Category.code == StockCategory.category_code
        ).filter(
            Issue.code == StockCategory.stock_code
        ).filter(
            Category.label == '地域'
        ).filter(
            Issue.A004_上市日期.isnot(None)
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'asof_date', 'region']
        return df


def normalized_thx_industry_data():
    """股票所处同花顺行业数据"""
    with session_scope() as sess:
        query = sess.query(
            StockCategory.stock_code,
            Issue.A004_上市日期,
            Category.title,
        ).filter(
            Category.code == StockCategory.category_code
        ).filter(
            Issue.code == StockCategory.stock_code
        ).filter(
            Category.label == '同花顺行业'
        ).filter(
            Issue.A004_上市日期.isnot(None)
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'asof_date', 'ths_industry']
        return df


def get_concept_categories():
    """概念类别映射{代码:名称}"""
    with session_scope() as sess:
        query = sess.query(
            Category.code,
            Category.title,
        ).filter(
            Category.label == '概念'
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['code', 'name']
        df.sort_values('code', inplace=True)
        return df.set_index('code').to_dict()['name']


def field_code_concept_maps():
    """
    概念映射

    Returns
    -------
    res ： 元组
        第一项：原始概念编码 -> 数据集字段编码（新编码）
        第二项：数据集字段编码 -> 概念名称

    Example
    -------
    第一项：{'00010002': 'A001', '00010003': 'A002', '00010004': 'A003', ...
    第二项：{'A001': '参股金融', 'A002': '可转债', 'A003': '上证红利'...

    """
    vs = get_concept_categories()
    no, key = pd.factorize(list(vs.keys()), sort=True)
    id_maps = {v: 'A{}'.format(str(k+1).zfill(3)) for k, v in zip(no, key)}
    name_maps = {v: vs[k] for (k, v) in id_maps.items()}
    return id_maps, name_maps


def handle_concept_data():
    """
    处理股票概念数据，保存列编码与列名称对照表
    """
    id_maps, name_maps = field_code_concept_maps()
    with session_scope() as sess:
        query = sess.query(
            StockCategory.stock_code,
            StockCategory.category_code,
        ).filter(
            Category.code == StockCategory.category_code
        ).filter(
            Category.label == '概念'
        ).filter(
            ~StockCategory.stock_code.startswith('9')
        ).filter(
            ~StockCategory.stock_code.startswith('2')
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'concept']
        out = pd.pivot_table(df,
                             values='concept',
                             index='sid',
                             columns='concept',
                             aggfunc=np.count_nonzero,
                             fill_value=0)
        out.rename(columns=id_maps, inplace=True)
        out = out.astype('bool')
        ipo = sess.query(
            Issue.code,
            Issue.A004_上市日期
        ).filter(
            Issue.A004_上市日期.isnot(None)
        ).all()
        ipo = pd.DataFrame.from_records(ipo)
        ipo.columns = ['sid', 'asof_date']
        ipo.set_index('sid', inplace=True)
    processed = pd.merge(ipo, out, left_index=True, right_index=True).reset_index()
    out = _normalize_ad_ts_sid(processed, 0)
    table_name = 'concepts'
    map_file = os.path.join(data_root('bcolz'), 'concept_maps.json')
    # 转换为bcolz格式并存储
    rootdir = _bcolz_table_path(table_name)
    if os.path.exists(rootdir):
        rmtree(rootdir)
    if os.path.exists(map_file):
        os.remove(map_file)
    odo(out, rootdir)
    with open(map_file,'w',encoding='utf-8') as file:
        json.dump(name_maps,file)   
    logger.info('表:{}，数据存储路径：{}'.format(table_name, rootdir))


def normalized_financial_data(df, maps):
    """财务报告数据"""
    df.drop(['更新时间', '序号', '公告日期'], axis=1, inplace=True)
    df.rename(columns={'股票代码': 'code', '报告日期': 'date'}, inplace=True)
    df.rename(columns={v: k for k, v in maps.items()}, inplace=True)
    return df.fillna(value=0.0)


def _factory(table_name):
    if table_name == 'adjustments':
        return normalized_dividend_data
    elif table_name == 'issues':
        return normalized_ipo_data
    elif table_name == 'short_names':
        return normalized_short_names_data
    elif table_name == 'special_treatments':
        return normalized_special_treatments_data        
    elif table_name == 'margins':
        return normalized_margins_data
    elif table_name == 'balance_sheets':
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

def _handle_factory(table_name):
    if table_name == 'concepts':
        handle_concept_data()

def _df_factory(table_name):
    if table_name == 'regions':
        return normalized_region_data()
    elif table_name == 'thx_industries':
        return normalized_thx_industry_data()
    elif table_name == 'csrc_industries':
        return normalized_thx_industry_data()


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
    rootdir = _bcolz_table_path(table_name)
    if os.path.exists(rootdir):
        rmtree(rootdir)
    odo(out, rootdir)
    logger.info('表:{}，数据存储路径：{}'.format(table_name, rootdir))


def _write_by_dataframe(df, table_name):
    """以bcolz格式写入数据框"""
    # 列名称及类型标准化
    out = _normalize_ad_ts_sid(df, ndays=0)
    # 转换为bcolz格式并存储
    rootdir = _bcolz_table_path(table_name)
    if os.path.exists(rootdir):
        rmtree(rootdir)
    odo(out, rootdir)
    logger.info('表:{}，数据存储路径：{}'.format(table_name, rootdir))


def convert_sql_data_to_bcolz(tables=None):
    """
    将sql数据表以bcolz格式存储，提高数据集加载速度

    Notes:
    ------
        写入财务报告数据时，默认财务报告公告日期为报告期后45天
    """
    to_does = tables if tables else INCLUDE_TABLES
    to_does = ensure_list(to_does)
    for table in to_does:
        logger.info('预处理表：{}'.format(table))
        if table in PS_TABLES:
            df = _df_factory(table)
            _write_by_dataframe(df, table)
        elif table in INCLUDE_TABLES:
            expr = STOCK_DB[table]
            ndays = 45 if table in QUARTERLY_TABLES else 0
            _write_by_expr(expr, ndays)
        elif table in HANDLE_TABLES:
            _handle_factory(table)
        else:
            raise ValueError('不支持数据表"{}"的转换'.format(table))
