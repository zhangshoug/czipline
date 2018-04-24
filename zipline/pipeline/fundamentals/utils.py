from collections import OrderedDict
from datashape import var, Record, Option
import pandas as pd
import numpy as np

from datashape import (
    Date,
    DateTime,
    Option,
    String,
    boolean,
    integral
)

from zipline.utils.numpy_utils import (default_missing_value_for_dtype, int64_dtype,
                                       datetime64ns_dtype)

#from zipline.utils.input_validation import expect_element

from ..common import (
    AD_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)


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
    """为表达式生成各字段的缺省默认值"""
    missing_values = {}
    for name, type_ in expr.dshape.measure.fields:
        # 可选项目，需要使用选项内部类型
        if isinstance(type_, Option):
            from_t = type_.ty
        else:
            from_t = type_

        if isinstance(from_t, Date):
            missing_values[name] = default_missing_value_for_dtype(
                datetime64ns_dtype)
        elif isinstance(from_t, DateTime):
            missing_values[name] = default_missing_value_for_dtype(
                datetime64ns_dtype)
        elif isinstance(from_t, String):
            missing_values[name] = 'unknown'
        elif from_t in boolean:
            missing_values[name] = False
        #elif from_t in integral:
        elif from_t is int64_dtype:
            missing_values[name] = -1
        else:
            missing_values[name] = np.nan
    return missing_values


def make_default_missing_values_for_df(dtypes):
    """DataFrame对象各字段生成缺省默认值"""
    missing_values = {}
    # 此处name为字段名称
    for f_name, type_ in dtypes.items():
        name = type_.name
        if name.startswith('int'):
            missing_values[f_name] = 0
        elif name.startswith('object'):
            missing_values[f_name] = 'unknown'
        else:
            missing_values[f_name] = default_missing_value_for_dtype(type_)
    return missing_values
