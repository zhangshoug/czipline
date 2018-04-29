from sqlalchemy import func
import pandas as pd
import numpy as np
from cswd.sql.base import session_scope
from cswd.sql.models import (Issue, StockDaily, Adjustment, DealDetail,
                             SpecialTreatment, SpecialTreatmentType)

DAILY_COLS = ['symbol', 'date',
              'open', 'high', 'low', 'close',
              'prev_close', 'change_pct',
              'volume', 'amount', 'turnover', 'cmv', 'tmv']
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
MINUTELY_COLS = ['symbol', 'date'] + OHLCV_COLS

ADJUSTMENT_COLS = ['symbol', 'date', 'amount', 'ratio',
                   'record_date', 'pay_date', 'listing_date']


def get_exchange(code):
    if code[0] in ('0', '3'):
        return "SZSE"
    else:
        return "SSE"


def get_start_dates():
    """
    股票上市日期

    Examples
    --------
    >>> df = get_start_dates()
    >>> df.head()
    symbol  start_date
    0  000001  1991-04-03
    1  000002  1991-01-29
    2  000003  1991-01-14
    3  000004  1991-01-14
    4  000005  1990-12-10
    """
    col_names = ['symbol', 'start_date']
    with session_scope() as sess:
        query = sess.query(
            Issue.code,
            Issue.A004_上市日期
        ).filter(
            Issue.A004_上市日期.isnot(None)
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = col_names
        return df


def get_end_dates():
    """
    股票结束日期。限定退市或者当前处于暂停上市状态的股票

    Examples
    --------
    >>> df = get_end_dates()
    >>> df.head()
    symbol    end_date
    0  000003  2002-06-14
    1  000013  2004-09-20
    2  000015  2001-10-22
    3  000024  2015-12-30
    4  000033  2017-07-07  
    """
    col_names = ['symbol', 'end_date']
    with session_scope() as sess:
        query = sess.query(
            SpecialTreatment.code,
            func.max(SpecialTreatment.date)
        ).group_by(
            SpecialTreatment.code
        ).having(
            SpecialTreatment.treatment.in_(
                [SpecialTreatmentType.delisting, SpecialTreatmentType.PT]
            )
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = col_names
        return df


def get_latest_short_name():
    """
    获取股票最新股票简称

    Examples
    --------
    >>> df = get_end_dates()
    >>> df.head()
         symbol  asset_name
    0     000001    平安银行
    1     000002    万 科Ａ
    2     000003   PT金田Ａ
    3     000004    国农科技
    4     000005    世纪星源     
    """
    col_names = ['symbol', 'asset_name']
    with session_scope() as sess:
        query = sess.query(
            StockDaily.code,
            StockDaily.A001_名称
        ).group_by(
            StockDaily.code
        ).having(
            func.max(StockDaily.date)
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = col_names
        return df


def gen_asset_metadata(only_in=True):
    """
    生成股票元数据

    Paras
    -----
    only_in : bool
        是否仅仅包含当前在市的股票，默认为真。

    Examples
    --------
    >>> df = gen_asset_metadata()
    >>> df.head()
    symbol asset_name first_traded last_traded exchange auto_close_date  \
    0  000001       平安银行   1991-01-02  2018-04-19     SZSE      2018-04-20
    1  000002       万 科Ａ   1991-01-02  2018-04-19     SZSE      2018-04-20
    2  000004       国农科技   1991-01-02  2018-04-19     SZSE      2018-04-20
    3  000005       世纪星源   1991-01-02  2018-04-19     SZSE      2018-04-20
    4  000006       深振业Ａ   1992-04-27  2018-04-19     SZSE      2018-04-20
    start_date    end_date
    0  1991-04-03  2018-04-19
    1  1991-01-29  2018-04-19
    2  1991-01-14  2018-04-19
    3  1990-12-10  2018-04-19
    4  1992-04-27  2018-04-19      
    """
    columns = ['symbol', 'first_traded', 'last_traded']
    with session_scope() as sess:
        query = sess.query(
            StockDaily.code,
            func.min(StockDaily.date),
            func.max(StockDaily.date)
        ).filter(
            ~StockDaily.code.startswith('2')
        ).filter(
            ~StockDaily.code.startswith('9')
        ).group_by(
            StockDaily.code
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = columns
        df['exchange'] = df['symbol'].map(get_exchange)
        df['auto_close_date'] = df['last_traded'].map(
            lambda x: x + pd.Timedelta(days=1))
        latest_name = get_latest_short_name()
        start_dates = get_start_dates()
        end_dates = get_end_dates()
        df = df.merge(
            latest_name, 'left', on='symbol'
        ).merge(
            start_dates, 'left', on='symbol'
        ).merge(
            end_dates, 'left', on='symbol'
        )
        # 对于未退市的结束日期，以最后交易日期代替
        df.loc[df.end_date.isna(), 'end_date'] = df.loc[df.end_date.isna(),
                                                        'last_traded']
        if only_in:
            df = df[~df.symbol.isin(end_dates.symbol)]
            df.reset_index(inplace=True, drop=True)
        return df


def _fill_zero(df):
    """填充因为停牌ohlc可能存在的0值"""
    # 将close放在第一列
    ohlc_cols = ['close', 'open', 'high', 'low']
    ohlc = df[ohlc_cols].copy()
    ohlc.replace(0.0, np.nan, inplace=True)
    ohlc.close.fillna(method='ffill', inplace=True)
    # 按列填充
    ohlc.fillna(method='ffill', axis=1, inplace=True)
    for col in ohlc_cols:
        df[col] = ohlc[col]
    return df


def fetch_single_equity(stock_code, start, end):
    """
    从本地数据库读取股票期间日线交易数据

    注
    --
    1. 除OHLCV外，还包括涨跌幅、成交额、换手率、流通市值、总市值、流通股本、总股本
    2. 使用bcolz格式写入时，由于涨跌幅存在负数，必须剔除该列！！！

    Parameters
    ----------
    stock_code : str
        要获取数据的股票代码
    start_date : datetime-like
        自开始日期(包含该日)
    end_date : datetime-like
        至结束日期

    return
    ----------
    DataFrame: OHLCV列的DataFrame对象。

    Examples
    --------
    >>> symbol = '000333'
    >>> start_date = '2017-4-1'
    >>> end_date = pd.Timestamp('2018-4-16')
    >>> df = fetch_single_equity(symbol, start_date, end_date)
    >>> df.iloc[:,:8]
        symbol        date   open   high    low  close  prev_close  change_pct
    0  000333  2018-04-02  53.30  55.00  52.68  52.84       54.53     -3.0992
    1  000333  2018-04-03  52.69  53.63  52.18  52.52       52.84     -0.6056
    2  000333  2018-04-04  52.82  54.10  52.06  53.01       52.52      0.9330
    3  000333  2018-04-09  52.91  53.31  51.00  51.30       53.01     -3.2258
    4  000333  2018-04-10  51.45  52.80  51.18  52.77       51.30      2.8655
    5  000333  2018-04-11  52.78  53.63  52.41  52.98       52.77      0.3980
    6  000333  2018-04-12  52.91  52.94  51.84  51.87       52.98     -2.0951
    7  000333  2018-04-13  52.40  52.47  51.01  51.32       51.87     -1.0603
    8  000333  2018-04-16  51.31  51.80  49.15  49.79       51.32     -2.9813    
    """
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    with session_scope() as sess:
        query = sess.query(
            StockDaily.code,
            StockDaily.date,
            StockDaily.A002_开盘价,
            StockDaily.A003_最高价,
            StockDaily.A004_最低价,
            StockDaily.A005_收盘价,
            StockDaily.A009_前收盘,
            StockDaily.A011_涨跌幅,
            StockDaily.A006_成交量,
            StockDaily.A007_成交金额,
            StockDaily.A008_换手率,
            StockDaily.A013_流通市值,
            StockDaily.A012_总市值
        ).filter(
            StockDaily.code == stock_code,
            StockDaily.date.between(start, end)
        )
        df = pd.DataFrame.from_records(query.all())
        df.columns = DAILY_COLS
        df = _fill_zero(df)
        df['circulating_share'] = df.cmv / df.close
        df['total_share'] = df.tmv / df.close
        return df


def _handle_minutely_data(df, exclude_lunch):
    """
    完成单个日期股票分钟级别数据处理
    """
    dts = pd.to_datetime(df[1].map(str) + ' ' + df[2])
    ohlcv = pd.Series(data=df[3].values, index=dts).resample('T').ohlc()
    ohlcv.fillna(method='ffill', inplace=True)
    # 成交量原始数据单位为手，换为股
    volumes = pd.Series(data=df[4].values, index=dts).resample('T').sum() * 100
    ohlcv.insert(4, 'volume', volumes)
    if exclude_lunch:
        # 默认包含上下界
        # 与交易日历保持一致，自31分开始
        pre = ohlcv.between_time('9:25', '9:31')

        def key(x): return x.date()
        grouped = pre.groupby(key)
        opens = grouped['open'].first()
        highs = grouped['high'].max()
        lows = grouped['low'].min()  # 考虑是否存在零值？
        closes = grouped['close'].last()
        volumes = grouped['volume'].sum()
        index = pd.to_datetime([str(x) + ' 9:31' for x in opens.index])
        add = pd.DataFrame({'open': opens.values,
                            'high': highs.values,
                            'low': lows.values,
                            'close': closes.values,
                            'volume': volumes.values
                            },
                           index=index)
        am = ohlcv.between_time('9:32', '11:30')
        pm = ohlcv.between_time('13:00', '15:00')
        return pd.concat([add, am, pm])
    else:
        return ohlcv


def fetch_single_minutely_equity(stock_code, start, end, exclude_lunch=True):
    """
    从本地数据库读取单个股票期间分钟级别交易明细数据

    注
    --
    1. 仅包含OHLCV列
    2. 原始数据按分钟进行汇总，first(open),last(close),max(high),min(low),sum(volume)

    Parameters
    ----------
    stock_code : str
        要获取数据的股票代码
    start_date : datetime-like
        自开始日期(包含该日)
    end_date : datetime-like
        至结束日期
    exclude_lunch ： bool
        是否排除午休时间，默认”是“

    return
    ----------
    DataFrame: OHLCV列的DataFrame对象。

    Examples
    --------
    >>> symbol = '000333'
    >>> start_date = '2018-4-1'
    >>> end_date = pd.Timestamp('2018-4-19')
    >>> df = fetch_single_minutely_equity(symbol, start_date, end_date)
    >>> df.tail()
                        close   high    low   open  volume
    2018-04-19 14:56:00  51.55  51.56  51.50  51.55  376400
    2018-04-19 14:57:00  51.55  51.55  51.55  51.55   20000
    2018-04-19 14:58:00  51.55  51.55  51.55  51.55       0
    2018-04-19 14:59:00  51.55  51.55  51.55  51.55       0
    2018-04-19 15:00:00  51.57  51.57  51.57  51.57  353900
    """
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    with session_scope() as sess:
        query = sess.query(
            DealDetail.code,
            DealDetail.date,
            DealDetail.A001_时间,
            DealDetail.A002_价格,
            DealDetail.A004_成交量
        ).filter(
            DealDetail.code == stock_code,
            DealDetail.date.between(start, end)
        )
        df = pd.DataFrame.from_records(query.all())
        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLS)
        return _handle_minutely_data(df, exclude_lunch)


def fetch_single_quity_adjustments(stock_code, start, end):
    """
    从本地数据库读取股票期间分红派息数据

    Parameters
    ----------
    stock_code : str
        要获取数据的股票代码
    start : datetime-like
        自开始日期
    end : datetime-like
        至结束日期

    return
    ----------
    DataFrame对象

    Examples
    --------
    >>> fetch_single_quity_adjustments('600000', '2010-4-1', '2018-4-16')
        symbol        date  amount  ratio record_date    pay_date listing_date
    0  600000  2010-06-10   0.150    0.3  2010-06-09  2010-06-11   2010-06-10
    1  600000  2011-06-03   0.160    0.3  2011-06-02  2011-06-07   2011-06-03
    2  600000  2012-06-26   0.300    0.0  2012-06-25  2012-06-26   2012-06-26
    3  600000  2013-06-03   0.550    0.0  2013-05-31  2013-06-03   2013-06-03
    4  600000  2014-06-24   0.660    0.0  2014-06-23  2014-06-24   2014-06-24
    5  600000  2015-06-23   0.757    0.0  2015-06-19  2015-06-23   2015-06-23
    6  600000  2016-06-23   0.515    0.1  2016-06-22  2016-06-24   2016-06-23
    7  600000  2017-05-25   0.200    0.3  2017-05-24  2017-05-26   2017-05-25
    """
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    with session_scope() as sess:
        query = sess.query(Adjustment.code,
                           Adjustment.date,
                           Adjustment.A002_派息,
                           Adjustment.A003_送股,
                           Adjustment.A004_股权登记日,
                           Adjustment.A005_除权基准日,
                           Adjustment.A006_红股上市日)
        query = query.filter(Adjustment.code == stock_code)
        query = query.filter(Adjustment.date.between(start, end))
        df = pd.DataFrame.from_records(query.all())
        if df.empty:
            # 返回一个空表
            return pd.DataFrame(columns=ADJUSTMENT_COLS)
        df.columns = ADJUSTMENT_COLS
        return df
