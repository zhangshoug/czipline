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


def _compute(factor_df, pct_df):
    """
    辅助计算

    参数
    ----
    factor_df : 因子数据框
    pct_df : 所有股票收益率数据框

    返回
    ----
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
    # 存储beta值(len(assets) * 11)
    beta_df = pd.DataFrame(index=asset_index, columns=SECTOR_MAPS.keys())
    # 主题因子
    fs_df = pd.DataFrame(index=asset_index, columns=STYLE_COLUMNS)
    # 部门收益率残差
    epsilon_sect = pd.Series(index=asset_index)
    # 部门回归及辅助计算
    for asset, sector_code in factor_df['sector'].items():
        index_code = SECTOR_INDEX_MAPS[int(sector_code)]
        # 重整指数收益率
        index_ret = get_cn_benchmark_returns(index_code)

        f = index_ret.reindex(date_index).fillna(0.0)
        r = pct_df[asset]
        k = SECTOR_MAPS[int(sector_code)]
        # 计算股票动量指标
        momentum = np.cumprod(r.iloc[m_start_loc:m_end_loc] + 1)[-1]

        # 计算股票收益率波动指标
        volatility = r.iloc[v_start_loc:v_end_loc].std()

        # 部门回归
        beta_vec, err = sector_linregress(k, f.values, r.values)
        epsilon_sect.loc[asset] = err
        beta_df.loc[asset, :] = beta_vec
        fs_df.loc[asset, 'momentum'] = momentum
        fs_df.loc[asset, 'volatility'] = volatility
        logger.info('日期：{} 股票:{}'.format(date_index[-1].strftime('%Y-%m-%d'),
                                         asset.symbol))
    # Z分数
    fs_df.loc[asset_index, 'size'] = factor_df.loc[asset_index, 'size']
    fs_df.loc[asset_index, 'value'] = factor_df.loc[asset_index, 'value']
    fs_df.loc[asset_index, 'reversal'] = factor_df.loc[asset_index, 'reversal']
    z_df = (fs_df - fs_df.mean()) / fs_df.std()
    f_style = []
    epsilon_style = []
    y = np.array(epsilon_sect.values)

    # 使用主题因子回归部门收益率残差
    for c in STYLE_COLUMNS:
        x = np.array(z_df[c].values)
        slope = stats.linregress(x, y)[0]
        err = y - slope * x
        epsilon_style.append(err)
        print(slope)
        # f_style.append(slope)
    # print(f_style)
    return fs_df, epsilon_sect


def make_pipeline():
    """部门编码及主题指标"""
    t_stocks = USEquityPricing.volume.latest >= 200000000.
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
    fs_df, errors = _compute(context.pipeline_results, rets)
    # print(fs_df.tail())


def before_trading_start(context, data):
    # 获取pipeline运行结果
    results = pipeline_output('factors')
    t = context.get_datetime('Asia/Shanghai')

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
