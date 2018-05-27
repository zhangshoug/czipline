"""
TODO：待完成！需要确定算法

运行该脚本必须满足：
    1. 完成sql数据库更新；
    2. 完成ingest；
    3. 完成sql-to-bcolz脚本；

运行频率：
    每个交易日晚上11：00运行

"""

import os
import sys

import logbook
import pandas as pd
from zipline import get_calendar
from zipline.data.benchmarks_cn import get_cn_benchmark_returns
from zipline.data.treasuries_cn import get_treasury_data
from zipline.pipeline import CustomFactor, Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.fundamentals import Fundamentals
from zipline.pipeline.builtin import AnnualFinancalData
from zipline.research import run_pipeline

from cswd.common.utils import data_root

# 设置显示日志
logbook.set_datetime_format('local')
logbook.StreamHandler(sys.stdout).push_application()
logger = logbook.Logger('构建ff因子')

calendar = get_calendar('SZSH')

all_trading_days = calendar.schedule.index
all_trading_days = all_trading_days[
    all_trading_days <= calendar.actual_last_session]

# 每月交易天数（近似值20，不同于美国股市，A股每年交易天数大约为244天）
normal_days = 31
business_days = int(0.66 * normal_days)


def get_rm_rf(earliest_date, symbol='000300'):
    """
    Rm-Rf(市场收益 - 无风险收益)
    基准股票指数收益率 - 国库券1个月收益率
    
    输出pd.Series(日期为Index), 'Mkt-RF', 'RF'二元组
    """
    start = '1990-1-1'
    end = pd.Timestamp('today')
    benchmark_returns = get_cn_benchmark_returns(symbol).loc[earliest_date:]
    treasury_returns = get_treasury_data(start, end)['1month'][earliest_date:]
    # 补齐缺省值
    treasury_returns = treasury_returns.reindex(
        benchmark_returns.index, method='ffill')
    return benchmark_returns, treasury_returns


class Returns(CustomFactor):
    """
    每个交易日每个股票窗口长度"business_days"期间收益率
    """
    window_length = business_days
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, price):
        out[:] = (price[-1] - price[0]) / price[0]


class PriorReturns(CustomFactor):
    """
    每个交易日每个股票窗口长度250的先前收益率
    
    p[-21] / p[-250] - 1
    """
    window_length = 250
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, price):
        out[:] = (price[-21] - price[0]) / price[0]


class MarketEquity(CustomFactor):
    """
    每个交易日每只股票所对应的总市值(取期初数)
    """
    window_length = business_days
    inputs = [USEquityPricing.tmv]

    def compute(self, today, assets, out, mcap):
        out[:] = mcap[0]


class BookEquity(CustomFactor):
    """
    每个交易日每只股票所对应的账面价值（所有者权益）(取期初数)
    """
    window_length = business_days
    inputs = [Fundamentals.balance_sheet.A107]

    def compute(self, today, assets, out, book):
        out[:] = book[0]


def make_pipeline():
    """构造3因子pipeline"""
    mkt_cap = MarketEquity()
    book_equity = BookEquity()
    returns = Returns()
    be_me = book_equity / mkt_cap
    # 去年营业利润
    t_1_operating_profit = AnnualFinancalData(inputs=[
        Fundamentals.profit_statement_yearly.A033,
        Fundamentals.profit_statement_yearly.report_end_date
    ])
    # 前年所有者权益
    t_2_equity = AnnualFinancalData(
        inputs=[
            Fundamentals.balance_sheet_yearly.A107,
            Fundamentals.balance_sheet_yearly.report_end_date
        ],
        t_n=2,
        window_lenght=500,
    )
    # 去年总资产
    t_1_total_assets = AnnualFinancalData(
        inputs=[
            Fundamentals.balance_sheet_yearly.A052,
            Fundamentals.balance_sheet_yearly.report_end_date
        ], )
    # 前年总资产
    t_2_total_assets = AnnualFinancalData(
        inputs=[
            Fundamentals.balance_sheet_yearly.A052,
            Fundamentals.balance_sheet_yearly.report_end_date
        ],
        t_n=2,
        window_lenght=500,
    )
    return Pipeline(
        columns={
            'market_cap': mkt_cap,
            'be_me': be_me,
            'returns': returns,
            'prior_returns': PriorReturns(),
            'op': t_1_operating_profit / t_2_equity,
            'inv': (t_1_total_assets - t_2_total_assets) / t_2_total_assets
        })


def compute_factors(one_day):
    """
    根据每天运行的pipeline结果，近似计算Fama-French 因子
    """
    factors = run_pipeline(make_pipeline(), one_day, one_day)
    # 获取后续使用的值
    returns = factors['returns']
    mkt_cap = factors.sort_values(['market_cap'], ascending=True)
    be_me = factors.sort_values(['be_me'], ascending=True)
    prior_returns = factors['prior_returns']

    # 根据市值划分总体，构造6个资产组合
    half = int(len(mkt_cap) * 0.5)
    small_caps = mkt_cap[:half]
    big_caps = mkt_cap[half:]

    thirty = int(len(be_me) * 0.3)
    seventy = int(len(be_me) * 0.7)
    growth = be_me[:thirty]
    neutral = be_me[thirty:seventy]
    value = be_me[seventy:]

    small_value = small_caps.index.intersection(value.index)
    small_neutral = small_caps.index.intersection(neutral.index)
    small_growth = small_caps.index.intersection(growth.index)

    big_value = big_caps.index.intersection(value.index)
    big_neutral = big_caps.index.intersection(neutral.index)
    big_growth = big_caps.index.intersection(growth.index)

    # 假设等权分配资金，取其投资组合回报率平均值
    sv = returns[small_value].mean()
    sn = returns[small_neutral].mean()
    sg = returns[small_growth].mean()

    bv = returns[big_value].mean()
    bn = returns[big_neutral].mean()
    bg = returns[big_growth].mean()

    # 按降序排列
    # 小市值组合收益率（序列没有列可以指定)
    s_p = prior_returns.loc[small_caps.index].sort_values(ascending=False)

    # 大市值组合收益率
    b_p = prior_returns.loc[big_caps.index].sort_values(ascending=False)

    shp = s_p.iloc[:int(len(s_p) * 0.3)].mean()  # 小市值组合收益率前70%的均值
    bhp = b_p.iloc[:int(len(b_p) * 0.3)].mean()  # 大市值组合收益率前70%的均值
    slp = s_p.iloc[int(len(s_p) * 0.7):].mean()  # 小市值组合收益率后30%的均值
    blp = b_p.iloc[int(len(b_p) * 0.7):].mean()  # 大市值组合收益率后30%的均值

    # 计算 SMB
    smb = (sv + sn + sg) / 3 - (bv + bn + bg) / 3

    # 计算 HML
    hml = (sv + bv) / 2 - (sg + bg) / 2

    # 计算 MOM
    mom = (shp + bhp) / 2 - (slp + blp) / 2

    return smb, hml, mom


def get_factors_data(dates, symbol):
    """计算指定日期列表"""
    start = dates[0]
    benchmark_returns, treasury_returns = get_rm_rf(start, symbol)
    res = []
    for d in dates:
        smb, hml, mom = compute_factors(d)
        mkt_rf = benchmark_returns.loc[d]
        rf = treasury_returns.loc[d]
        res.append((d, smb, hml, mom, mkt_rf, rf))
        logger.info('当前处理日期：{}'.format(d.date()))
    df = pd.DataFrame.from_records(
        res, columns=['date', 'SMB', 'HML', 'Mom', 'Mkt-RF', 'RF'])
    df.set_index('date', inplace=True)
    return df


class FFFactor(object):
    # 沪深300指数自2002-01-07开始
    # 国库券数据1个月资金成本自2013-4-26开始
    # 数据最早开始日期设定为2013-5-1之后的最早交易日期
    earliest_date = pd.Timestamp('2013-05-01').tz_localize('UTC')

    def __init__(self, benchmark_symbol='000300', file_path=None):
        if file_path:
            self.file_path = file_path
        else:
            self.file_path = os.path.join(data_root('ff_factors'), 'ff.pkl')
        self.benchmark_symbol = benchmark_symbol

    def _read(self):
        try:
            return pd.read_pickle(self.file_path)
        except FileNotFoundError:
            return None

    def refresh(self):
        local_df = self._read()
        if local_df is None:
            dates = all_trading_days[all_trading_days > self.earliest_date]
            data = get_factors_data(dates, self.benchmark_symbol)
            data.to_pickle(self.file_path)
        else:
            last_date = local_df.index[-1]
            if last_date < all_trading_days[-1]:
                dates = all_trading_days[all_trading_days > last_date]
                data = get_factors_data(dates, self.benchmark_symbol)
                # 防止出现刷新时将异常数据存入！
                data.dropna(inplace=True)
                data = pd.concat([local_df, data])
                data.to_pickle(self.file_path)

    @property
    def data(self):
        logger.info('正在整理三因子数据，请耐心等待......')
        self.refresh()
        return self._read()


def main():
    ff = FFFactor()
    ff.refresh()


if __name__ == '__main__':
    main()