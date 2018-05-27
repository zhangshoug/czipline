"""
部门编码
同花顺行业 -> 部门代码
"""
MARKET_MAPS = {
    9: '上海B股',
    6: '上海主板',
    3: '创业板',
    2: '深圳B股',
    1: '中小板',
    0: '深圳主板',
    -1: '未知',
}

SUPER_SECTOR_NAMES = {
    1: '周期',
    2: '防御',
    3: '敏感',
}

SECTOR_NAMES = {
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

QUARTERLY_TABLES = [
    'balance_sheets', 'profit_statements', 'cashflow_statements', 'chnls',
    'cznls', 'ylnls', 'yynls', 'zyzbs'
]