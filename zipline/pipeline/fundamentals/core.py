import re
import pandas as pd
import json
import os
from cswd.common.utils import data_root
from zipline.utils.memoize import classlazyval
from cswd.sql.constants import (
    MARGIN_MAPS,
    BALANCESHEET_ITEM_MAPS,
    PROFITSTATEMENT_ITEM_MAPS,
    CASHFLOWSTATEMENT_ITEM_MAPS,
    ZYZB_ITEM_MAPS,
    YLNL_ITEM_MAPS,
    CHNL_ITEM_MAPS,
    CZNL_ITEM_MAPS,
    YYNL_ITEM_MAPS,
)

from ..data.dataset import BoundColumn

# from .categories import (get_report_item_categories,
#                          report_item_meta,
#                          get_concept_categories)

from .normalize import (normalized_stock_region_data,
                        normalized_cninfo_industry_data,
                        normalized_csrc_industry_data,
                        normalized_special_treatment_data,
                        normalized_short_name_data,
                        normalized_time_to_market_data,
                        from_bcolz_data)

ITEM_CODE_PATTERN = re.compile(r'A\d{3}')

_SUPER_SECTOR_NAMES = {
    1: '周期',
    2: '防御',
    3: '敏感',
}

_SECTOR_NAMES = {
    101: '基本材料',
    102: '主要消费',
    103: '金融服务',
    104: '房地产',
    205: '可选消费',
    206: '医疗保健',
    207: '公用事业',
    308: '通讯服务',
    309: '能源',
    310: '工业领域',
    311: '工程技术',
}


def verify_code(code):
    if not re.match(ITEM_CODE_PATTERN, code):
        raise ValueError('编码格式应为A+三位整数。输入{}'.format(code))


class Fundamentals(object):
    """股票基础数据集容器类"""
    @staticmethod
    def has_column(column):
        """简单判定列是否存在于`Fundamentals`各数据集中"""
        return type(column) == BoundColumn

    @staticmethod
    def query_margin_name(code):
        """查询融资融券列名称（输入编码）"""
        verify_code(code)
        return MARGIN_MAPS[code]

    @classlazyval
    def margin(self):
        """股票融资融券数据集"""
        return from_bcolz_data(table_name='margins')

    @staticmethod
    def query_balancesheet_name(code):
        """查询资产负债表所对应的科目名称（输入编码）"""
        verify_code(code)
        return BALANCESHEET_ITEM_MAPS[code]

    @staticmethod
    def query_balancesheet_code(key):
        """根据名称关键词模糊查询资产负债表项目编码"""
        out = {k: v for k, v in BALANCESHEET_ITEM_MAPS.items() if key in v}
        return out

    @classlazyval
    def balance_sheet(self):
        """资产负债数据集"""
        return from_bcolz_data(table_name='balance_sheets')

    @staticmethod
    def query_profit_name(code):
        """查询利润表所对应的科目名称（输入编码）"""
        verify_code(code)
        return PROFITSTATEMENT_ITEM_MAPS[code]

    @staticmethod
    def query_profit_code(key):
        """根据名称关键词模糊查询利润表项目编码"""
        out = {k: v for k, v in PROFITSTATEMENT_ITEM_MAPS.items() if key in v}
        return out

    @classlazyval
    def profit_statement(self):
        """利润表数据集"""
        return from_bcolz_data(table_name='profit_statements')

    @staticmethod
    def query_cashflow_name(code):
        """查询现金流量表列名称（输入编码）"""
        verify_code(code)
        return CASHFLOWSTATEMENT_ITEM_MAPS[code]

    @staticmethod
    def query_cashflow_code(key):
        """根据名称关键词模糊查询现金流量表项目编码"""
        out = {k: v for k, v in CASHFLOWSTATEMENT_ITEM_MAPS.items()
               if key in v}
        return out

    @classlazyval
    def cash_flow(self):
        """现金流量表数据集"""
        return from_bcolz_data(table_name='cashflow_statements')

    @staticmethod
    def query_key_financial_indicator_name(code):
        """查询主要财务指标列名称（输入编码）"""
        verify_code(code)
        return ZYZB_ITEM_MAPS[code]

    @staticmethod
    def query_key_financial_indicator_code(key):
        """根据名称关键词模糊查询主要财务指标表项目编码"""
        out = {k: v for k, v in ZYZB_ITEM_MAPS.items() if key in v}
        return out

    @classlazyval
    def key_financial_indicator(self):
        """主要财务指标数据集"""
        return from_bcolz_data(table_name='zyzbs')

    @staticmethod
    def query_earnings_ratio_name(code):
        """查询盈利能力指标列名称（输入编码）"""
        verify_code(code)
        return YLNL_ITEM_MAPS[code]

    @staticmethod
    def query_earnings_ratio_code(key):
        """根据名称关键词模糊查询盈利能力指标项目编码"""
        out = {k: v for k, v in YLNL_ITEM_MAPS.items() if key in v}
        return out

    @classlazyval
    def earnings_ratio(self):
        """盈利能力指标数据集"""
        return from_bcolz_data(table_name='ylnls')

    @staticmethod
    def query_solvency_ratio_name(code):
        """查询偿还能力指标名称（输入编码）"""
        verify_code(code)
        return CHNL_ITEM_MAPS[code]

    @staticmethod
    def query_solvency_ratio_code(key):
        """根据名称关键词模糊查询偿还能力指标项目编码"""
        out = {k: v for k, v in CHNL_ITEM_MAPS.items() if key in v}
        return out

    @classlazyval
    def solvency_ratio(self):
        """偿还能力指标数据集"""
        return from_bcolz_data(table_name='chnls')

    @staticmethod
    def query_growth_ratio_name(code):
        """查询成长能力指标名称（输入编码）"""
        verify_code(code)
        return CZNL_ITEM_MAPS[code]

    @staticmethod
    def query_growth_ratio_code(key):
        """根据名称关键词模糊查询成长能力指标项目编码"""
        out = {k: v for k, v in CZNL_ITEM_MAPS.items() if key in v}
        return out

    @classlazyval
    def growth_ratio(self):
        """成长能力指标数据集"""
        return from_bcolz_data(table_name='cznls')

    @staticmethod
    def query_operation_ratio_name(code):
        """查询营运能力指标名称（输入编码）"""
        verify_code(code)
        return YYNL_ITEM_MAPS[code]

    @staticmethod
    def query_operation_ratio_code(key):
        """根据名称关键词模糊查询营运能力指标项目编码"""
        out = {k: v for k, v in YYNL_ITEM_MAPS.items() if key in v}
        return out

    @classlazyval
    def operation_ratio(self):
        """营运能力指标数据集"""
        return from_bcolz_data(table_name='yynls')

    @staticmethod
    def query_supper_sector_name(code):
        """查询超级部门行业名称（输入编码）"""
        return _SUPER_SECTOR_NAMES[code]

    @staticmethod
    def query_sector_name(code):
        """查询部门行业名称（输入编码）"""
        return _SECTOR_NAMES[code]

    @staticmethod
    def query_concept_name(code):
        """查询股票概念中文名称（输入字段编码）"""
        code = code.upper()
        if not re.match(ITEM_CODE_PATTERN, code):
            raise ValueError('概念编码由"A"与三位数字编码组合而成。如A021')
        map_file = os.path.join(data_root('bcolz'), 'concept_maps.json')
        with open(map_file,'r',encoding='utf-8') as file:
            name_maps = json.load(file)   
        return name_maps[code]

    @staticmethod
    def query_concept_code(key):
        """模糊查询概念编码（输入概念关键词）"""
        map_file = os.path.join(data_root('bcolz'), 'concept_maps.json')
        with open(map_file,'r',encoding='utf-8') as file:
            name_maps = json.load(file)  
        out = {k: v for k, v in name_maps.items() if key in v}        
        return out

    @classlazyval
    def concept(self):
        """股票概念数据集"""
        return from_bcolz_data(table_name='concepts')
        
    # -------------以下部分为单列------------------
    @classlazyval
    def ipo(self):
        """上市日期（单列）"""
        return from_bcolz_data(table_name='issues').ipo

    @classlazyval
    def short_name(self):
        """股票简称（单列）"""
        return from_bcolz_data(table_name='short_names').short_name

    @classlazyval
    def region(self):
        """股票所属地区（单列）"""
        return from_bcolz_data(table_name='regions').region

    @classlazyval
    def ths_industry(self):
        """股票所属同花顺行业（单列）"""
        return from_bcolz_data(table_name='ths_industries').ths_industry

    @classlazyval
    def csrc_industry(self):
        """股票所属证监会行业（单列）"""
        return from_bcolz_data(table_name='csrc_industries').csrc_industry

    @classlazyval
    def treatment(self):
        """股票特别处理（单列）"""
        return normalized_special_treatment_data().treatment

    @classlazyval
    def annual_dividend(self):
        """年度股利（单列）"""
        return from_bcolz_data(table_name='adjustments').amount


