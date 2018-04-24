"""
规范sql存储的数据，转换适用于from_blaze格式，生成DataSet对象

如果数据量大，以bcolz格式（文件）存储；否则直接转换为DataFramde（内存中）

转换后的数据格式为：
asof_date    timestamp    sid      int_value  str_value date_value  float_value
2014-01-06   2014-01-07   1        0          A         2001-01-01  2000.30
2014-01-07   2014-01-08   333      1          B         2010-02-03  0.003
2014-01-08   2014-01-09   2024     2          C         2017-11-01  3000.11
2014-01-09   2014-01-10   300001   2          D         2017-01-20  200.20

如`f（sid,asof_date）`非标量，需要转换为不同列名称，或者生成多个数据集。
确保`f（sid,asof_date）`为标量

如股票概念原始数据：

    一只股票在一个时点可能存在多个概念：
        asof_date    sid      concept_id  
        2014-01-06   1        20001   
        2014-01-06   1        20002 

    转换为多列数据集：
        asof_date    sid   C20001    C20002   C20003
        2014-01-06   1      True       True    False

行业分类数据则根据编制部门转换为巨潮行业、证监会行业二个数据集

特别注意：
    from_blaze将原始数据中的NaN视为不存在数据，则会提取最接近当前时点的数据
    需要根据实际情形处理好原始数据中的Nan
    如财务报告

"""
from functools import lru_cache
import re
import pandas as pd
import blaze
import bcolz
from collections import OrderedDict
from datashape import var, Record, Option
from odo import discover, odo

from cswd.common.utils import ensure_list

from ..loaders.blaze import global_loader, from_blaze

from ..common import (
    AD_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)

from .utils import (_normalized_dshape, 
                    _normalize_ad_ts_sid,
                    make_default_missing_values_for_df,
                    make_default_missing_values_for_expr,
                    )

from .ctable import _bcolz_table_path
from .categories import get_industry_categories, field_code_concept_maps
from .base import STOCK_DB


INDEX_COLS = [SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME]

CNINFO_ID_PATTERN = re.compile(r'C\d{6}$')
CSRC_ID_PATTERN = re.compile(r'\w\d{2}$')


def make_dataset(df, name):
    """构造指定数据集名称的数据集"""
    old_dshape = discover(df)
    expr = blaze.data(df, _normalized_dshape(old_dshape), name)
    return from_blaze(expr,
                      loader = global_loader,
                      no_deltas_rule = 'ignore',
                      no_checkpoints_rule = 'ignore',
                      missing_values = make_default_missing_values_for_df(df.dtypes))


def normalized_time_to_market_data():
    """上市日期"""
    e = STOCK_DB.stock_codes
    df = odo(e[['code', 'time_to_market']], pd.DataFrame)
    df['date'] = df['time_to_market']
    return make_dataset(_normalize_ad_ts_sid(df), 'time_to_market')


def normalized_stock_region_data():
    """所处地域"""
    e0 = STOCK_DB.regions.drop_field('last_updated')
    e1 = STOCK_DB.region_stocks.drop_field('last_updated')
    e = blaze.join(e1, e0, 'region_id', 'id').drop_field('region_id')
    df = _normalize_ad_ts_sid(odo(e, pd.DataFrame))
    return make_dataset(df, 'region')


def get_supper_sector_code(industry_code):
    """映射为超级行业分类代码"""
    code = industry_code[:3]
    if code in ('C02','C04','C07'):
        return 1
    if code in ('C05','C06','C10'):
        return 2
    if code in ('C01','C03','C08','C09'):
        return 3
    return -1


def get_sector_code(industry_code):
    """映射为部门行业分类代码"""
    if industry_code[:3] == 'C01':
        return 309
    if industry_code[:3] == 'C02':
        return 101
    if industry_code[:3] == 'C03':
        return 310
    if industry_code[:3] == 'C04':
        return 102
    if industry_code[:3] == 'C05':
        return 205
    if industry_code[:3] == 'C06':
        return 206
    if industry_code.startswith('C07'):
        if industry_code[:5] == 'C0703':
            return 104
        else:
            return 103
    if industry_code[:3] == 'C08':
        return 311
    if industry_code[:3] == 'C09':
        return 308
    if industry_code[:3] == 'C10':
        return 207
    return -1


def get_cninfo_industry_data():
    maps = get_industry_categories()
    e = STOCK_DB.industry_stocks.drop_field('last_updated')
    df = _normalize_ad_ts_sid(odo(e, pd.DataFrame))
    ## 保留('sid','industry_id)组合的唯一值
    df.drop_duplicates(['sid','industry_id'], inplace = True)

    res = df.loc[df.industry_id.str.match(CNINFO_ID_PATTERN), :].copy()
    res.loc[:,'industry'] = res['industry_id'].map(maps['cninfo'])
    res.loc[:,'group'] = [maps['cninfo'][x[:5]] for x in res.industry_id]
    res.loc[:,'sector'] = res.industry_id.map(get_sector_code)
    res.loc[:,'supper_sector'] = res.industry_id.map(get_supper_sector_code)
    res.drop('industry_id', axis=1, inplace=True)
    return res


def normalized_cninfo_industry_data():
    """国证行业"""
    res = get_cninfo_industry_data()
    return make_dataset(res, 'cninfo')


def normalized_csrc_industry_data():
    """证监会行业
    注：
        证监会行业分二级，大类及子类
        分类标准与超级部门分类不一致，无法映射
    """
    maps = get_industry_categories()
    e = STOCK_DB.industry_stocks.drop_field('last_updated')
    df = _normalize_ad_ts_sid(odo(e, pd.DataFrame))
    ## 保留('sid','industry_id)组合的唯一值
    df.drop_duplicates(['sid','industry_id'], inplace = True)

    res = df.loc[df.industry_id.str.match(CSRC_ID_PATTERN), :].copy()
    res.loc[:,'industry'] = res['industry_id'].map(maps['csrc'])
    res.loc[:,'sector'] = [maps['csrc'][x[:1]] for x in res.industry_id]
    res.drop('industry_id', axis=1, inplace=True)
    return make_dataset(res, 'csrc')


def normalized_special_treatment_data():
    """特别处理数据集（如ST）"""
    e = STOCK_DB.special_treatments.drop_field('last_updated')
    df = _normalize_ad_ts_sid(odo(e, pd.DataFrame))
    return make_dataset(df, 'treatment')


def normalized_short_name_data():
    """股票简称数据集"""
    e = STOCK_DB.short_names.drop_field('last_updated')
    df = _normalize_ad_ts_sid(odo(e, pd.DataFrame))
    df.fillna('none', inplace=True)
    return make_dataset(df, 'short_name')


@lru_cache(None)
def from_bcolz_data(table_name, columns = None, expr_name = None):
    """读取bcolz格式的数据，生成DataSet"""
    rootdir = _bcolz_table_path(table_name)
    ctable = bcolz.ctable(rootdir=rootdir, mode='r')
    if columns:
        columns = ensure_list(columns)
        sub_ctable = ctable[columns]
    else:
        sub_ctable = ctable
    df = sub_ctable.todataframe()
    raw_dshape = discover(df)
    df_dshape = _normalized_dshape(raw_dshape)
    if expr_name is None:
        expr = blaze.data(df, name = table_name, dshape = df_dshape)
    else:
        expr = blaze.data(df, name = expr_name, dshape = df_dshape)
    return from_blaze(expr,
                      loader = global_loader,
                      no_deltas_rule = 'ignore',
                      no_checkpoints_rule = 'ignore',
                      missing_values = make_default_missing_values_for_expr(expr))