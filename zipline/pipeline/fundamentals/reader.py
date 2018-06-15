import re
import warnings

import bcolz
import pandas as pd

import blaze
from cswd.sql.constants import (
    BALANCESHEET_ITEM_MAPS, CASHFLOWSTATEMENT_ITEM_MAPS, CHNL_ITEM_MAPS,
    CZNL_ITEM_MAPS, MARGIN_MAPS, PROFITSTATEMENT_ITEM_MAPS, YLNL_ITEM_MAPS,
    YYNL_ITEM_MAPS, ZYZB_ITEM_MAPS)
from odo import discover
from zipline.utils.memoize import classlazyval

from ..data.dataset import BoundColumn
from ..loaders.blaze import from_blaze, global_loader
from .base import bcolz_table_path
from .constants import (MARKET_MAPS, QUARTERLY_TABLES, SECTOR_NAMES,
                        SUPER_SECTOR_NAMES)
from .utils import _normalized_dshape, make_default_missing_values_for_expr

ITEM_CODE_PATTERN = re.compile(r'A\d{3}')
warnings.filterwarnings("ignore")


def verify_code(code):
    if not re.match(ITEM_CODE_PATTERN, code):
        raise ValueError('编码格式应为A+三位整数。输入{}'.format(code))


def from_bcolz_data(table_name, yearly=False):
    """读取bcolz格式的数据，生成DataSet"""
    rootdir = bcolz_table_path(table_name)
    ctable = bcolz.ctable(rootdir=rootdir, mode='r')
    df = ctable.todataframe()
    if yearly:
        msg = '年报数据仅支持表名称{}'.format(QUARTERLY_TABLES)
        assert table_name in QUARTERLY_TABLES, msg
        # 刷选出年报部分(仅适用于财务报告)
        is_year_end = df.report_end_date.map(lambda x: x.is_year_end)
        df = df[is_year_end]
    raw_dshape = discover(df)
    df_dshape = _normalized_dshape(raw_dshape)
    expr = blaze.data(df, name=table_name, dshape=df_dshape)
    return from_blaze(
        expr,
        loader=global_loader,
        no_deltas_rule='ignore',
        no_checkpoints_rule='ignore',
        missing_values=make_default_missing_values_for_expr(expr))


def query_maps(table_name, attr_name, key_to_int=False):
    """查询bcolz表中属性值"""
    rootdir = bcolz_table_path(table_name)
    ct = bcolz.open(rootdir)
    d = ct.attrs[attr_name]
    if key_to_int:
        return {int(k): v for k, v in d.items()}
    else:
        return d


class Fundamentals(object):
    """股票基础数据集容器类"""

    @staticmethod
    def has_column(column):
        """简单判定列是否存在于`Fundamentals`各数据集中"""
        return type(column) == BoundColumn

    #=======================根据数据集列编码查询列名称=================#
    # 数据集列使用四位编码，首位为A，其余按原始序列连续编号
    # 要查询列所代表的具体含义，使用下面方法
    # 输入A001之类的编码
    @staticmethod
    def margin_col_name(code):
        """查询融资融券列名称（输入编码）"""
        verify_code(code)
        return MARGIN_MAPS[code]

    @staticmethod
    def balancesheet_col_name(code):
        """查询资产负债表所对应的科目名称（输入编码）"""
        verify_code(code)
        return BALANCESHEET_ITEM_MAPS[code]

    @staticmethod
    def profit_col_name(code):
        """查询利润表所对应的科目名称（输入编码）"""
        verify_code(code)
        return PROFITSTATEMENT_ITEM_MAPS[code]

    @staticmethod
    def cashflow_col_name(code):
        """查询现金流量表列名称（输入编码）"""
        verify_code(code)
        return CASHFLOWSTATEMENT_ITEM_MAPS[code]

    @staticmethod
    def key_financial_indicator_col_name(code):
        """查询主要财务指标列名称（输入编码）"""
        verify_code(code)
        return ZYZB_ITEM_MAPS[code]

    @staticmethod
    def earnings_ratio_col_name(code):
        """查询盈利能力指标列名称（输入编码）"""
        verify_code(code)
        return YLNL_ITEM_MAPS[code]

    @staticmethod
    def solvency_ratio_col_name(code):
        """查询偿还能力指标名称（输入编码）"""
        verify_code(code)
        return CHNL_ITEM_MAPS[code]

    @staticmethod
    def growth_ratio_col_name(code):
        """查询成长能力指标名称（输入编码）"""
        verify_code(code)
        return CZNL_ITEM_MAPS[code]

    @staticmethod
    def operation_ratio_col_name(code):
        """查询营运能力指标名称（输入编码）"""
        verify_code(code)
        return YYNL_ITEM_MAPS[code]
    
    @classlazyval
    def concept_maps(self):
        return query_maps('infoes', 'concept')

    @staticmethod
    def concept_col_name(code):
        """股票概念中文名称（输入编码）"""
        table_name = 'infoes'
        attr_name = 'concept'
        maps = query_maps(table_name, attr_name)
        return maps[code]

    #===================根据数据集列名称关键字模糊查询列编码===============#
    # 有时需要根据列关键字查找列编码，使用下面方法
    # 输入关键字查询列编码
    @staticmethod
    def balancesheet_col_code(key):
        """根据名称关键词模糊查询资产负债表项目编码"""
        out = {k: v for k, v in BALANCESHEET_ITEM_MAPS.items() if key in v}
        return out

    @staticmethod
    def profit_col_code(key):
        """根据名称关键词模糊查询利润表项目编码"""
        out = {k: v for k, v in PROFITSTATEMENT_ITEM_MAPS.items() if key in v}
        return out

    @staticmethod
    def cashflow_col_code(key):
        """根据名称关键词模糊查询现金流量表项目编码"""
        out = {
            k: v
            for k, v in CASHFLOWSTATEMENT_ITEM_MAPS.items() if key in v
        }
        return out

    @staticmethod
    def key_financial_indicator_col_code(key):
        """根据名称关键词模糊查询主要财务指标表项目编码"""
        out = {k: v for k, v in ZYZB_ITEM_MAPS.items() if key in v}
        return out

    @staticmethod
    def earnings_ratio_col_code(key):
        """根据名称关键词模糊查询盈利能力指标项目编码"""
        out = {k: v for k, v in YLNL_ITEM_MAPS.items() if key in v}
        return out

    @staticmethod
    def solvency_ratio_col_code(key):
        """根据名称关键词模糊查询偿还能力指标项目编码"""
        out = {k: v for k, v in CHNL_ITEM_MAPS.items() if key in v}
        return out

    @staticmethod
    def growth_ratio_col_code(key):
        """根据名称关键词模糊查询成长能力指标项目编码"""
        out = {k: v for k, v in CZNL_ITEM_MAPS.items() if key in v}
        return out

    @staticmethod
    def operation_ratio_col_code(key):
        """根据名称关键词模糊查询营运能力指标项目编码"""
        out = {k: v for k, v in YYNL_ITEM_MAPS.items() if key in v}
        return out

    @staticmethod
    def concept_col_code(key):
        """模糊查询概念编码（输入概念关键词）"""
        table_name = 'infoes'
        attr_name = 'concept'
        maps = query_maps(table_name, attr_name)
        out = {k: v for k, v in maps.items() if key in v}
        return out

    #========================编码中文含义========================#
    # 为提高读写速度，文本及类别更改为整数编码，查找整数所代表的含义，使用
    # 下列方法。数字自0开始，长度为len(类别)
    # 输入数字（如触发Keyerror，请减少数值再次尝试

    @classlazyval
    def supper_sector_maps(self):
        return query_maps('infoes', 'super_sector_code', True)

    @staticmethod
    def supper_sector_cname(code):
        """超级部门编码含义"""
        code = int(code)
        table_name = 'infoes'
        attr_name = 'super_sector_code'
        maps = query_maps(table_name, attr_name, True)
        return maps[code]

    @classlazyval
    def sector_maps(self):
        return query_maps('infoes', 'sector_code', True)

    @staticmethod
    def sector_cname(code):
        """部门编码含义"""
        code = int(code)
        table_name = 'infoes'
        attr_name = 'sector_code'
        maps = query_maps(table_name, attr_name, True)
        return maps[code]

    @staticmethod
    def sector_code(key):
        """关键词查询地域编码"""
        table_name = 'infoes'
        attr_name = 'sector_code'
        maps = query_maps(table_name, attr_name)
        out = {k: v for k, v in maps.items() if key in v}
        return out

    @staticmethod
    def market_cname(code):
        """市场版块编码含义"""
        code = str(code)
        table_name = 'infoes'
        attr_name = 'market'
        maps = query_maps(table_name, attr_name)
        return maps[code]

    @classlazyval
    def region_maps(self):
        return query_maps('infoes', 'region')

    @staticmethod
    def region_cname(code):
        """地域版块编码含义"""
        code = str(code)
        table_name = 'infoes'
        attr_name = 'region'
        maps = query_maps(table_name, attr_name)
        return maps[code]

    @staticmethod
    def region_code(key):
        """关键词查询地域编码"""
        table_name = 'infoes'
        attr_name = 'region'
        maps = query_maps(table_name, attr_name)
        out = {k: v for k, v in maps.items() if key in v}
        return out

    @classlazyval
    def csrc_industry_maps(self):
        return query_maps('infoes', 'csrc_industry')

    @staticmethod
    def csrc_industry_cname(code):
        """证监会行业编码含义"""
        code = str(code)
        table_name = 'infoes'
        attr_name = 'csrc_industry'
        maps = query_maps(table_name, attr_name)
        return maps[code]

    @staticmethod
    def csrc_industry_code(key):
        """关键词模糊查询证监会行业编码"""
        table_name = 'infoes'
        attr_name = 'csrc_industry'
        maps = query_maps(table_name, attr_name)
        out = {k: v for k, v in maps.items() if key in v}
        return out

    @classlazyval
    def ths_industry_maps(self):
        return query_maps('infoes', 'ths_industry')

    @staticmethod
    def ths_industry_cname(code):
        """同花顺行业编码含义"""
        code = str(code)
        table_name = 'infoes'
        attr_name = 'ths_industry'
        maps = query_maps(table_name, attr_name)
        return maps[code]

    @staticmethod
    def ths_industry_code(key):
        """关键词模糊查询同花顺行业编码"""
        table_name = 'infoes'
        attr_name = 'ths_industry'
        maps = query_maps(table_name, attr_name)
        out = {k: v for k, v in maps.items() if key in v}
        return out

    @classlazyval
    def cn_industry_maps(self):
        return query_maps('infoes', 'cn_industry')

    @staticmethod
    def cn_industry_cname(code):
        """国证行业编码含义"""
        code = str(code)
        table_name = 'infoes'
        attr_name = 'cn_industry'
        maps = query_maps(table_name, attr_name)
        return maps[code]

    @staticmethod
    def cn_industry_code(key):
        """关键词模糊查询国证行业编码"""
        table_name = 'infoes'
        attr_name = 'cn_industry'
        maps = query_maps(table_name, attr_name)
        out = {k: v for k, v in maps.items() if key in v}
        return out

    #========================数据集========================#

    @classlazyval
    def info(self):
        """股票静态信息数据集"""
        return from_bcolz_data(table_name='infoes')

    @classlazyval
    def balance_sheet(self):
        """资产负债数据集"""
        return from_bcolz_data(table_name='balance_sheets')

    @classlazyval
    def balance_sheet_yearly(self):
        """资产负债数据集(仅包含年度报告)"""
        return from_bcolz_data('balance_sheets', True)

    @classlazyval
    def profit_statement(self):
        """利润表数据集"""
        return from_bcolz_data(table_name='profit_statements')

    @classlazyval
    def profit_statement_yearly(self):
        """年度利润表数据集(仅包含年度报告)"""
        return from_bcolz_data('profit_statements', True)

    @classlazyval
    def cash_flow(self):
        """现金流量表数据集"""
        return from_bcolz_data(table_name='cashflow_statements')

    @classlazyval
    def cash_flow_yearly(self):
        """现金流量表数据集(仅包含年度报告)"""
        return from_bcolz_data('cashflow_statements', True)

    @classlazyval
    def key_financial_indicator(self):
        """主要财务指标数据集"""
        return from_bcolz_data(table_name='zyzbs')

    @classlazyval
    def key_financial_indicator_yearly(self):
        """主要财务指标数据集(仅包含年度报告)"""
        return from_bcolz_data('zyzbs', True)

    @classlazyval
    def earnings_ratio(self):
        """盈利能力指标数据集"""
        return from_bcolz_data(table_name='ylnls')

    @classlazyval
    def earnings_ratio_yearly(self):
        """盈利能力指标数据集(仅包含年度报告)"""
        return from_bcolz_data('ylnls', True)

    @classlazyval
    def solvency_ratio(self):
        """偿还能力指标数据集"""
        return from_bcolz_data(table_name='chnls')

    @classlazyval
    def solvency_ratio_yearly(self):
        """偿还能力指标数据集(仅包含年度报告)"""
        return from_bcolz_data('chnls', True)

    @classlazyval
    def growth_ratio(self):
        """成长能力指标数据集"""
        return from_bcolz_data(table_name='cznls')

    @classlazyval
    def growth_ratio_yearly(self):
        """成长能力指标数据集(仅包含年度报告)"""
        return from_bcolz_data('cznls', True)

    @classlazyval
    def operation_ratio(self):
        """营运能力指标数据集"""
        return from_bcolz_data(table_name='yynls')

    @classlazyval
    def operation_ratio_yearly(self):
        """营运能力指标数据集(仅包含年度报告)"""
        return from_bcolz_data('yynls', True)

    @classlazyval
    def margin(self):
        """融资融券数据集"""
        return from_bcolz_data(table_name='margins')

    @classlazyval
    def dividend(self):
        """每股股利数据集"""
        return from_bcolz_data(table_name='adjustments')

    #========================单列========================#
    @classlazyval
    def rating(self):
        """股票评级（单列）"""
        return from_bcolz_data(table_name='rating').rating

    @classlazyval
    def short_name(self):
        """股票简称（单列）"""
        return from_bcolz_data(table_name='short_names').short_name

    @classlazyval
    def treatment(self):
        """股票特别处理（单列）"""
        return from_bcolz_data(table_name='special_treatments').treatment
