import os
from shutil import rmtree
from logbook import Logger
from odo import odo
import bcolz

from .base import bcolz_table_path
from .utils import _normalize_ad_ts_sid


def write_dataframe(df, table_name, ndays, attr_dict):
    """以bcolz格式写入数据框"""
    logger = Logger(table_name)
    # 列名称及类型标准化
    out = _normalize_ad_ts_sid(df, ndays=ndays)
    # 转换为bcolz格式并存储
    rootdir = bcolz_table_path(table_name)
    if os.path.exists(rootdir):
        rmtree(rootdir)
    odo(out, rootdir)
    logger.info('数据存储路径：{}'.format(rootdir))
    # 设置属性
    ct = bcolz.open(rootdir)
    for k, v in attr_dict.items():
        ct.attrs[k] = v
