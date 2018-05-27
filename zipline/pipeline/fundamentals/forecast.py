"""
股票相关预测数据

包括：
    1. 股票评级
    2. 

有关业绩预测，数据中大量存在空值，暂不纳入。

处理方式：
    1. 按股票、日期分组，简单平均
"""

import pandas as pd
from logbook import Logger

from cswd.tasks.sina_data_center import DataCenter
from .writer import write_dataframe

RATING_SCORE = {
    '买入':5,
    '持有':4,
    '中性':3,
    '减持':2,
    '卖出':1,
}

def rating_table():
    """获取整理后的股票评级表"""
    is_a = lambda x:x[0] in ('0','3','6')
    rating_data = DataCenter('rating').data
    # 仅需A股
    rating_data = rating_data[rating_data['股票代码'].map(is_a)]
    df = pd.DataFrame(
        {
            'sid':rating_data['股票代码'],
            'timestamp':pd.to_datetime(rating_data['评级日期']),
            'rating': rating_data['最新评级'].map(RATING_SCORE).values,
        }
    )
    res = df.groupby(['sid','timestamp']).mean().reset_index()
    res['asof_date'] = res['timestamp'] - pd.Timedelta(days=1)
    return res

def write_rating_to_bcolz():
    table_name = 'rating'
    logger = Logger(table_name)
    logger.info('准备数据......')
    df = rating_table()
    attr_dict = {}
    ndays = 0
    write_dataframe(df, table_name, ndays, attr_dict)