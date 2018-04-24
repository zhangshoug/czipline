import pandas as pd
from odo import odo
from functools import lru_cache

# from cswd.sql.db_schema import report_item_meta

from .base import STOCK_DB

# 地区
@lru_cache()
def get_region_categories():
    """地区类别映射"""
    expr = STOCK_DB.regions.drop_field('last_updated')
    df = odo(expr, pd.DataFrame)
    return df.set_index('id').to_dict()['name']

# 行业（证监会、巨潮）
@lru_cache()
def get_industry_categories():
    """按编制部门输出{代码:名称}映射"""
    expr = STOCK_DB.industries.drop_field('last_updated')
    df = odo(expr, pd.DataFrame)
    res = {}
    for name, group in df.groupby('department'):
        res[name] = group.set_index('industry_id').to_dict()['name']
    return res

# 概念
@lru_cache()
def get_concept_categories():
    """概念类别映射{代码:名称}"""
    expr = STOCK_DB.concepts.drop_field('last_updated')
    df = odo(expr, pd.DataFrame)
    return df.set_index('concept_id').to_dict()['name']

@lru_cache()
def field_code_concept_maps():
    """
    概念映射

    Returns
    -------
    res ： 元组
        第一项：原始概念编码 -> 数据集字段编码（新编码）
        第二项：数据集字段编码 -> 概念名称
    Notes
    -------
		DataSet.C001 - DataSet.CNNN

    Example
    -------
    第一项：{'021001': 'C176', '021002': 'C196'...
    第二项：{'C176': 'H股', 'C196': 'QFII持股', 'C129': 'ST'...

    """
    vs = get_concept_categories()
    no, key = pd.factorize(list(vs.keys()),sort=True)
    id_maps = {v:'C{}'.format(str(k+1).zfill(3)) for k,v in zip(no, key)}
    name_maps = {v:vs[k] for (k,v) in id_maps.items()}
    return id_maps, name_maps

# 财务科目
# @lru_cache()
# def get_report_item_categories():
#     """财务科目类别映射"""
#     name_s = report_item_meta[['code','name']]
#     name_s.set_index('code', inplace = True)
#     name_s = name_s['name']
#     return name_s.to_dict()
