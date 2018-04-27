from collections import OrderedDict

import numpy as np
import pandas as pd
from datashape import (Date, DateTime, Option, Record, String, boolean,
                       integral, var)

from zipline.utils.numpy_utils import (bool_dtype,
                                       object_dtype,
                                       int64_dtype,
                                       float64_dtype,
                                       datetime64ns_dtype,
                                       default_missing_value_for_dtype)

from ..common import AD_FIELD_NAME, SID_FIELD_NAME, TS_FIELD_NAME
from ..loaders.blaze.core import datashape_type_to_numpy


def _normalized_dshape(input_dshape):
    """关闭dshape中关键可选，保留原始数据类型"""
    fields = OrderedDict(input_dshape.measure.fields)
    out_dshape = []
    for name, type_ in fields.items():
        if name in (SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME):
            if isinstance(type_, Option):
                type_ = type_.ty
        out_dshape.append([name, type_])
    return var * Record(out_dshape)


def _normalize_ad_ts_sid(df, ndays=0):
    """通用转换

    code(str) -> sid(int64)
    date(date) -> asof_date(timestamp)
    date(date) + ndays -> timestamp(timestamp)

    """
    if 'asof_date' in df.columns:
        df['asof_date'] = pd.to_datetime(df['asof_date'])
    else:
        df['asof_date'] = pd.to_datetime(df['date'])
        df.drop('date', axis=1, inplace=True)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df['timestamp'] = df['asof_date'] + pd.Timedelta(days=ndays)

    if 'sid' in df.columns:
        df['sid'] = df['sid'].map(lambda x: int(x))
    else:
        df['sid'] = df['code'].map(lambda x: int(x))
        df.drop('code', axis=1, inplace=True)
    return df


def make_default_missing_values_for_expr(expr):
    """数据集输出时的字段缺省默认值"""
    missing_values = {}
    for name, type_ in expr.dshape.measure.fields:
        n_type = datashape_type_to_numpy(type_)
        if n_type is object_dtype:
            missing_values[name] = '未定义'
        elif n_type is bool_dtype:
            missing_values[name] = False
        elif n_type is int64_dtype:
            missing_values[name] = -1
        else:
            missing_values[name] = default_missing_value_for_dtype(n_type)
    return missing_values


def make_default_missing_values_for_df(dtypes):
    """DataFrame对象各字段生成缺省默认值"""
    missing_values = {}
    # 此处name为字段名称
    for f_name, type_ in dtypes.items():
        name = type_.name
        if name.startswith('int'):
            # # 应为0
            missing_values[f_name] = 0
        elif name.startswith('object'):
            missing_values[f_name] = 'unknown'
        else:
            missing_values[f_name] = default_missing_value_for_dtype(type_)
    return missing_values
