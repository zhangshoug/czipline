"""
TODO:去掉参数t
"""

import numpy as np
import pandas as pd

__all__ = [
    'check_series_or_dict', 'get_ix', 'time_matrix_locator', 'time_locator',
    'null_checker', 'non_null_data_args'
]


class NotFoundAsset(Exception):
    """找不到对应资产异常"""
    pass


def check_series_or_dict(x, name):
    """检查输入类型。输入若不是序列或者字典类型其中之一，则触发类型异常"""
    is_s = isinstance(x, pd.Series)
    is_d = isinstance(x, dict)
    if not (is_s or is_d):
        raise TypeError('参数名称{}类型错误。应为pd.Series或者dict'.format(name))


def get_ix(series, assets):
    """从指定序列中查找assets位置序号"""
    # assets必须可迭代
    index = series.index
    ix = index.get_indexer(assets)
    if -1 in ix:
        raise NotFoundAsset('无法找到资产：{}'.format(
            pd.Index(assets).difference(index)))
    return ix


def null_checker(obj):
    """Check if obj contains NaN."""
    if (isinstance(obj, pd.Panel) or isinstance(obj, pd.DataFrame)
            or isinstance(obj, pd.Series)):
        if np.any(pd.isnull(obj)):
            raise ValueError('Data object contains NaN values', obj)
    elif np.isscalar(obj):
        if np.isnan(obj):
            raise ValueError('Data object contains NaN values', obj)
    else:
        raise TypeError('Data object can only be scalar or Pandas.')


def non_null_data_args(f):
    def new_f(*args, **kwds):
        for el in args:
            null_checker(el)
        for el in kwds.values():
            null_checker(el)
        return f(*args, **kwds)

    return new_f


def time_matrix_locator(obj, t, as_numpy=False):
    """Retrieve a matrix from a time indexed Panel, or a static DataFrame."""
    if isinstance(obj, pd.Panel):
        res = obj.iloc[obj.axes[0].get_loc(t, method='pad')]
        return res.values if as_numpy else res
    elif isinstance(obj, pd.DataFrame):
        return obj.values if as_numpy else obj
    else:  # obj not pandas
        raise TypeError('Expected Pandas DataFrame or Panel, got:', obj)


def time_locator(obj, t, as_numpy=False):
    """Retrieve data from a time indexed DF, or a Series or scalar."""
    if isinstance(obj, pd.DataFrame):
        res = obj.iloc[obj.axes[0].get_loc(t, method='pad')]
        return res.values if as_numpy else res
    elif isinstance(obj, pd.Series):
        return obj.values if as_numpy else obj
    elif np.isscalar(obj):
        return obj
    else:
        raise TypeError('Expected Pandas DataFrame, Series, or scalar. Got:',
                        obj)
