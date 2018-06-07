"""
此模块内容要放入到～/pkg_source/czipline/zipline/pipeline/risk.py

且更改为pipeline
"""

import os
from functools import lru_cache

import logbook
import numpy as np
import pandas as pd
from scipy import stats

from zipline import run_algorithm
from zipline.api import attach_pipeline, get_datetime, pipeline_output
from zipline.data.benchmarks_cn import get_cn_benchmark_returns
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import RSI
from zipline.pipeline.fundamentals import Fundamentals
from zipline.research import to_tdates

logger = logbook.Logger('计算风险因子')
PPY = 244  # 每年交易天数
PPM = 21  # 每月交易天数
# 顺序再编码
SECTOR_MAPS = {
    101: 1,
    102: 2,
    103: 3,
    104: 4,
    205: 5,
    206: 6,
    207: 7,
    308: 8,
    309: 9,
    310: 10,
    311: 11
}
# 部门影射指数代码
# 把工程技术影射为信息技术
SECTOR_INDEX_MAPS = {
    101: '399614',
    102: '399617',
    103: '399619',
    104: '399241',
    205: '399616',
    206: '399618',
    207: '399622',
    308: '399621',
    309: '399613',
    310: '399615',
    311: '399620'
}

STYLE_COLUMNS = ['momentum', 'size', 'value', 'reversal', 'volatility']


def matrix_linregress(X, Y):
    """矩阵回归"""
    pass


def sector_linregress(k, x, y):
    """部门线性回归

    参数
    ----
    x:基准收益率
    y:股票收益率
    """
    slope = stats.linregress(x, y)[0]  # 只需要第一项斜率
    err = y - slope * x
    beta = np.zeros(len(SECTOR_MAPS))
    # 必须减去1(向量从0开始)
    beta[k - 1] = slope
    return beta, err[-1]  # 只需要最后一项


@lru_cache()
def _index_return():
    """
    与行业相关指数所有日期收益率表

    以行业keys为列
    """
    df = pd.DataFrame(columns=SECTOR_INDEX_MAPS.keys())
    for i, k in enumerate(SECTOR_INDEX_MAPS.keys()):
        temp = get_cn_benchmark_returns(SECTOR_INDEX_MAPS[k])
        if i == 0:
            df[k] = temp.values
        else:
            new_index = df.index.union(temp.index).unique()
            df = df.reindex(new_index).fillna(0.0)
            df[k] = temp.reindex(new_index).fillna(0.0)
    return df


def _compute_loading(factor_df, pct_df):
    """
    辅助计算

    参数
    ----
    factor_df : 因子数据框
    pct_df : 所有股票收益率数据框

    返回
    ----
    m个股票16因子列数据框
    """
    # 动量位置
    m_start_loc = -12 * PPM + 1
    m_end_loc = -PPM + 1
    # 波动位置
    v_start_loc = -6 * PPM + 1
    v_end_loc = -1
    # 股票Index
    asset_index = factor_df.index
    # 日期Index
    date_index = pct_df.index
    # 存储beta值(len(assets) * 11 + 5)
    beta_columns = list(SECTOR_MAPS.keys()) + STYLE_COLUMNS
    beta_df = pd.DataFrame(index=asset_index, columns=beta_columns)
    # 主题因子
    fs_df = pd.DataFrame(index=asset_index, columns=STYLE_COLUMNS)

    # 部门回归及辅助计算
    for asset, sector_code in factor_df['sector'].items():
        # 居然有股票没有部门代码的?暂且跳过，事后再检查
        if sector_code is np.nan:
            continue
        # 此处不适宜使用矩阵回归，需分资产处理
        f = _index_return().loc[date_index, int(sector_code)].fillna(0.0)
        r = pct_df[asset]
        k = SECTOR_MAPS[int(sector_code)]
        # 计算股票动量指标
        momentum = np.cumprod(r.iloc[m_start_loc:m_end_loc] + 1)[-1]

        # 计算股票收益率波动指标
        volatility = r.iloc[v_start_loc:v_end_loc].std()

        # 部门回归
        beta_vec, _ = sector_linregress(k, f.values, r.values)
        # epsilon_sect.loc[asset] = err
        beta_df.loc[asset, list(SECTOR_MAPS.keys())] = beta_vec
        fs_df.loc[asset, 'momentum'] = momentum
        fs_df.loc[asset, 'volatility'] = volatility
        logger.info('日期：{} 股票:{}'.format(date_index[-1].strftime(r'%Y-%m-%d'),
                                         asset.symbol))
    # Z分数
    fs_df.loc[asset_index, 'size'] = factor_df.loc[asset_index, 'size']
    fs_df.loc[asset_index, 'value'] = factor_df.loc[asset_index, 'value']
    fs_df.loc[asset_index, 'reversal'] = factor_df.loc[asset_index, 'reversal']
    z_df = (fs_df - fs_df.mean()) / fs_df.std()

    # 赋值
    beta_df.loc[asset_index, STYLE_COLUMNS] = z_df.loc[asset_index, STYLE_COLUMNS]

    return beta_df


def make_pipeline():
    """部门编码及主题指标"""
    # 这里主要目的是限制计算量，最终要改为0， 代表当天交易的股票
    t_stocks = USEquityPricing.volume.latest >= 50000000
    e = Fundamentals.balance_sheet.A107.latest
    tmv = USEquityPricing.tmv.latest
    return Pipeline(
        columns={
            'sector': Fundamentals.info.sector_code.latest,
            # 使用公司流通市值
            'size': USEquityPricing.cmv.latest,
            'value': e / tmv,
            'reversal': RSI()
        },
        screen=t_stocks)


def initialize(context):
    pipe = attach_pipeline(make_pipeline(), name='factors')


def handle_data(context, data):
    prices = data.history(
        context.stocks, fields="close", bar_count=2 * PPY + 1, frequency="1d")
    # 获取2年收益率数据
    rets = prices.pct_change(1).iloc[1:]
    beta_df = _compute_loading(context.pipeline_results, rets)
    t0 = rets.index[-1]
    t1 = context.get_datetime('Asia/Shanghai')
    print('回测日期', t1)
    print('计算日期',t0)
    print(beta_df.tail())


def before_trading_start(context, data):
    # 获取pipeline运行结果
    results = pipeline_output('factors')

    # 存储结果到上下文
    context.pipeline_results = results
    context.stocks = results.index  # 限定为当天交易的股票


def main():
    bundle = 'cndaily'
    start, end = '2018-6-1', '2018-6-4'
    _, start_date, end_date = to_tdates(start, end)
    run_algorithm(
        start_date,
        end_date,
        initialize,
        100000,
        handle_data=handle_data,
        before_trading_start=before_trading_start,
        bundle=bundle,
        bm_symbol='000300')


if __name__ == '__main__':
    main()
