"""
静态分类数据

股票地域、行业、概念、交易所等信息本不属于静态数据，但极少变动，视同静态数据。简化处理

基础分类信息
asof_date    timestamp    sid      地域  证监会行业    同花顺行业  部门编码  超级部门   A001
2014-01-06   2014-01-07   1        湖南         A         A2       311       3       0
2014-01-07   2014-01-08   333      广东         B         B2       307       3       1
2014-01-08   2014-01-09   2024     广西         C         C2       207       2       0
2014-01-09   2014-01-10   300001   北京         D         D2       102       1       1

注意：
    将文字转换为类别，将映射保存为属性，提高读写速度。
"""
import pandas as pd
import numpy as np
from logbook import Logger

from cswd.sql.base import session_scope
from cswd.sql.models import Stock, Issue, Category, StockCategory, TradingCalendar
from cswd.websource.juchao import fetch_industry_stocks
from .writer import write_dataframe
from .constants import SECTOR_NAMES, SUPER_SECTOR_NAMES


def latest_trading_day():
    """最近一个交易日期(不包含当日)"""
    today = pd.Timestamp('today').date()
    with session_scope() as sess:
        return sess.query(
            TradingCalendar.date,
        ).filter(
            TradingCalendar.date < today
        ).filter(
            TradingCalendar.is_trading == True
        ).order_by(
            TradingCalendar.date.desc()
        ).limit(1).scalar()


def market_map(stock_code):
    """
    股票代码映射市场版块
        '上海B股',
        '上海主板',
        '创业板',
        '深圳B股',
        '中小板',
        '深圳主板'
    """
    if stock_code[:3] == '002':
        return '中小板'
    elif stock_code[0] == '9':
        return '上海B股'
    elif stock_code[0] == '6':
        return '上海主板'
    elif stock_code[0] == '3':
        return '创业板'
    elif stock_code[0] == '2':
        return '深圳B股'
    elif stock_code[0] == '0':
        return '深圳主板'
    else:
        return '未知'


def sector_code_map(industry_code):
    """
    国证行业分类映射为部门行业分类
    
    国证一级行业分10类，转换为sector共11组，单列出房地产。
    """
    if industry_code[:3] == 'C01':
        return 309
    if industry_code[:3] == 'C02':
        return 101
    if industry_code[:3] == 'C03':
        return 310
    if industry_code[:3] == 'C04':
        return 205
    if industry_code[:3] == 'C05':
        return 102
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


def supper_sector_code_map(sector_code):
    """行业分类映射超级行业分类"""
    if sector_code == -1:
        return -1
    return int(str(sector_code)[0])


def ipo_info():
    """
    股票上市日期(A股，不含B股)
    """
    with session_scope() as sess:
        query = sess.query(
            Stock.code,
            Issue.A004_上市日期,
        ).filter(
            Stock.code == Issue.code
        ).filter(
            Issue.A004_上市日期.isnot(None)
        ).filter(
            ~Stock.code.startswith('2')
        ).filter(
            ~Stock.code.startswith('9')
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'asof_date']
        return df.sort_values('sid')


def handle_cate(df, col):
    """指定列更改为编码，输出更改后的表对象及类别映射"""
    c = df[col].astype('category')
    df[col] = c.cat.codes.astype('int64')
    maps = {k: v for k, v in enumerate(c.cat.categories)}
    return df, maps


def region_info():
    """股票所处地域"""
    with session_scope() as sess:
        query = sess.query(
            StockCategory.stock_code,
            Category.title,
        ).filter(
            Category.code == StockCategory.category_code
        ).filter(
            Category.label == '地域'
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'region']
        df.sort_values('sid', inplace=True)
        return df


def scrc_industry_info():
    """证监会行业"""
    with session_scope() as sess:
        query = sess.query(
            StockCategory.stock_code,
            Category.title,
        ).filter(
            Category.code == StockCategory.category_code
        ).filter(
            Category.label == '证监会行业'
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'csrc_industry']
        return df


def ths_industry_info():
    """同花顺行业"""
    with session_scope() as sess:
        query = sess.query(
            StockCategory.stock_code,
            Category.title,
        ).filter(
            Category.code == StockCategory.category_code
        ).filter(
            Category.label == '同花顺行业'
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'ths_industry']
        return df


def cninfo_industry_info():
    """
    获取上一个交易日最新的国证行业分类表，
    并映射部门及超级部门分类
    """
    d = latest_trading_day()
    df = fetch_industry_stocks(date_=d, department='cninfo')
    # 在此先映射部门及超级部门代码
    # 在转换为类别前，处理部门、超级部门编码
    # 为保持一致性，使用原始三位数编码
    df['sector_code'] = df.industry_code.map(sector_code_map)
    df['super_sector_code'] = df.sector_code.map(supper_sector_code_map)
    return pd.DataFrame(
        {
            'sid': df.code.values,
            'cn_industry': df.industry_name.values,
            'sector_code':df.sector_code.values,
            'super_sector_code':df.super_sector_code.values,
        }
    )


def concept_categories():
    """概念类别映射{代码:名称}"""
    with session_scope() as sess:
        query = sess.query(
            Category.code,
            Category.title,
        ).filter(
            Category.label == '概念'
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['code', 'name']
        df.sort_values('code', inplace=True)
        return df.set_index('code').to_dict()['name']


def field_code_concept_maps():
    """
    概念映射

    Returns
    -------
    res ： 元组
        第一项：原始概念编码 -> 数据集字段编码（新编码）
        第二项：数据集字段编码 -> 概念名称

    Example
    -------
    第一项：{'00010002': 'A001', '00010003': 'A002', '00010004': 'A003', ...
    第二项：{'A001': '参股金融', 'A002': '可转债', 'A003': '上证红利'...

    """
    vs = concept_categories()
    no, key = pd.factorize(list(vs.keys()), sort=True)
    id_maps = {v: 'A{}'.format(str(k+1).zfill(3)) for k, v in zip(no, key)}
    name_maps = {v: vs[k] for (k, v) in id_maps.items()}
    return id_maps, name_maps


def concept_info():
    """股票概念"""
    id_maps, _ = field_code_concept_maps()
    with session_scope() as sess:
        query = sess.query(
            StockCategory.stock_code,
            StockCategory.category_code,
        ).filter(
            Category.code == StockCategory.category_code
        ).filter(
            Category.label == '概念'
        ).filter(
            ~StockCategory.stock_code.startswith('9')
        ).filter(
            ~StockCategory.stock_code.startswith('2')
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['sid', 'concept']
        out = pd.pivot_table(df,
                             values='concept',
                             index='sid',
                             columns='concept',
                             aggfunc=np.count_nonzero,
                             fill_value=0)
        out.rename(columns=id_maps, inplace=True)
        return out.astype('bool').reset_index()


def _fillna(df, cate_cols):
    # 填充NA值
    values = {}
    bool_cols = df.columns[df.columns.str.match(r'A\d{3}')]
    for col in bool_cols:
        values[col] = False
    for col in cate_cols:
        values[col] = -1
    df.fillna(value=values, inplace=True)


def static_info_table():
    """静态分类信息合并表"""
    maps = {}
    _, name_maps = field_code_concept_maps()
    ipo = ipo_info()
    ipo['market'] = ipo.sid.map(market_map)
    region = region_info()
    scrc = scrc_industry_info()
    ths = ths_industry_info()
    cninfo = cninfo_industry_info()
    concept = concept_info()
    df = ipo.merge(
        region, how='left', on='sid'
    ).merge(
        scrc, how='left', on='sid'
    ).merge(
        ths, how='left', on='sid'
    ).merge(
        cninfo, how='left', on='sid'
    ).merge(
        concept, how='left', on='sid'
    )
    cate_cols = ['market', 'region', 'csrc_industry',
                 'ths_industry', 'cn_industry']
    for col in cate_cols:
        df, m = handle_cate(df, col)
        maps[col] = m
    maps['concept'] = name_maps
    maps['sector_code'] = SECTOR_NAMES
    maps['super_sector_code'] = SUPER_SECTOR_NAMES
    # 处理
    _fillna(df, cate_cols)
    return df, maps


def write_static_info_to_bcolz():
    table_name = 'infoes'
    logger = Logger(table_name)
    logger.info('准备数据......')
    ndays = 0
    df, attr_dict = static_info_table()
    # df['asof_date'] = pd.to_datetime(df['asof_date'].values).tz_localize(None)
    write_dataframe(df, table_name, ndays, attr_dict)
