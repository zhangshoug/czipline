from functools import lru_cache

import numpy as np
import pandas as pd
from numexpr import evaluate
from numpy import abs, average, clip, diff, dstack, inf
from scipy import stats

from zipline.data.benchmarks_cn import get_cn_benchmark_returns
from zipline.research import to_tdates
from zipline.utils.math_utils import (nanargmax, nanargmin, nanmax, nanmean,
                                      nanmin, nanstd)

from .data import USEquityPricing
from .factors import RSI, CustomFactor
from .factors.factor import zscore
from .factors.statistical import vectorized_beta
from .fundamentals import Fundamentals
from .pipeline import Pipeline
from .mixins import SingleInputMixin

__all__ = [
    'Momentum', 'Value', 'Size', 'ShortTermReversal', 'Volatility',
    'BasicMaterials', 'ConsumerCyclical', 'FinancialServices', 'RealEstate',
    'ConsumerDefensive', 'HealthCare', 'Utilities', 'CommunicationServices',
    'Energy', 'Industrials', 'Technology', 'risk_loading_pipeline'
]

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


@lru_cache()
def _index_return():
    """
    与行业相关指数所有日期收益率表

    以行业keys为列
    """
    dfs = []
    for k in SECTOR_INDEX_MAPS.keys():
        dfs.append(get_cn_benchmark_returns(SECTOR_INDEX_MAPS[k]))
    df = pd.concat(dfs, axis=1)
    df.columns = SECTOR_INDEX_MAPS.keys()
    return df.fillna(0.0)


class SectorBase(CustomFactor):
    """
    Risk Model loadings 基础类
    """
    inputs = [USEquityPricing.close, Fundamentals.info.sector_code]
    window_length = 2 * PPY + 1
    sector_code = 101

    def compute(self, today, assets, out, closes, scs):
        # 找出行业分类代码相应的资产序号
        s_code = scs[-1].astype(int)
        zero_locs = np.nonzero(s_code != self.sector_code)[0]
        locs = np.nonzero(s_code == self.sector_code)[0]
        # 所处行业为基本材料的股票收益率
        all_returns = np.diff(closes[:, locs], axis=0) / closes[1:, locs]

        # 行业指数收益率
        target_returns = _index_return().loc[:today, self.sector_code].iloc[
            -self.window_length + 1:].fillna(0.0).values
        shape = (self.window_length - 1, 1)  # 输入shape = (n,1)
        target_returns = target_returns.reshape(shape)

        allowed_missing_count = int(self.window_length * 0.25)
        temp = vectorized_beta(
            dependents=all_returns,
            independent=target_returns,
            allowed_missing=allowed_missing_count,
        )
        out[locs] = temp
        out[zero_locs] = 0.0  # 非该行业股票beta值，其数据设定为0.0


class BasicMaterials(SectorBase):
    """
    Risk Model loadings for the basic materials sector.
    """
    sector_code = 101


class ConsumerCyclical(SectorBase):
    """
    Risk Model loadings for the consumer cyclical sector.
    """
    sector_code = 102


class FinancialServices(SectorBase):
    """
    Risk Model loadings for the financial services sector.
    """
    sector_code = 103


class RealEstate(SectorBase):
    """
    Risk Model loadings for the real estate sector.
    """
    sector_code = 104


class ConsumerDefensive(SectorBase):
    """
    Risk Model loadings for the consumer defensive sector.
    """
    sector_code = 205


class HealthCare(SectorBase):
    """
    Risk Model loadings for the health care sector.
    """
    sector_code = 206


class Utilities(SectorBase):
    """
    Risk Model loadings for the utilities sector.
    """
    sector_code = 207


class CommunicationServices(SectorBase):
    """
    Risk Model loadings for the communication services sector.
    """
    sector_code = 308


class Energy(SectorBase):
    """
    Risk Model loadings for the communication energy sector.
    """
    sector_code = 309


class Industrials(SectorBase):
    """
    Risk Model loadings for the industrials sector.
    """
    sector_code = 310


class Technology(SectorBase):
    """
    Risk Model loadings for the technology sector.
    """
    sector_code = 311


class Momentum(CustomFactor):
    """
    Risk Model loadings for the “momentum” style factor.

    This factor captures differences in returns between stocks that 
    have had large gains in the last 11 months and stocks that have 
    had large losses in the last 11 months.
    """
    inputs = [USEquityPricing.close]
    window_length = 2 * PPY

    def compute(self, today, assets, out, closes):
        # TODO：改用Returns计算
        # 计算尾部11个月(前12个月到前1个月之间的11个月)的累计收益率
        last_11m_loc = int(self.window_length / 12 / 2 * 11)
        last_1m_loc = int(self.window_length / 12 / 2)
        ratio = (
            np.diff(closes, axis=0) / closes[1:])[-last_11m_loc:-last_1m_loc]
        out[:] = zscore(np.cumprod(1 + ratio, axis=0)[-1])


class Value(CustomFactor):
    """
    Quantopian Risk Model loadings for the “value” style factor.

    This factor captures differences in returns between “expensive” stocks 
    and “inexpensive” stocks, measured by the ratio between each stock’s book 
    value and its market cap.
    """
    inputs = [Fundamentals.balance_sheet.A107, USEquityPricing.tmv]
    window_length = 2

    def compute(self, today, assets, out, s, m):
        # 财务数据单位为万元
        out[:] = zscore(s[-1] * 10000 / m[-1])


class Size(CustomFactor):
    """
    Risk Model loadings for the “size” style factor.

    This factor captures difference in returns between stocks with 
    high market capitalizations and stocks with low market capitalizations.

    """
    inputs = [USEquityPricing.cmv]
    window_length = 2

    def compute(self, today, assets, out, cmv):
        out[:] = zscore(np.log10(cmv[-1]))


class ShortTermReversal(CustomFactor, SingleInputMixin):
    """
    Risk Model loadings for the “short term reversal” style factor.

    This factor captures differences in returns between stocks that have experienced 
    short term losses and stocks that have experienced short term gains.
    """
    window_length = 15
    inputs = (USEquityPricing.close, )
    window_safe = True

    def compute(self, today, assets, out, closes):
        diffs = diff(closes, axis=0)
        ups = nanmean(clip(diffs, 0, inf), axis=0)
        downs = abs(nanmean(clip(diffs, -inf, 0), axis=0))
        tmp = -evaluate(
            "100 - (100 / (1 + (ups / downs)))",
            local_dict={
                'ups': ups,
                'downs': downs
            },
            global_dict={},
        )
        out[:] = zscore(tmp)


class Volatility(CustomFactor):
    """
    Risk Model loadings for the “volatility” style factor.

    This factor captures differences in returns between stocks that experience 
    large price fluctuations and stocks that have relatively stable prices.
    """
    inputs = [USEquityPricing.close]
    window_length = 6 * PPM

    def compute(self, today, assets, out, closes):
        r = np.diff(closes, axis=0) / closes[1:]
        out[:] = zscore(r.std(axis=0))


def risk_loading_pipeline():
    """
    为风险模型创建一个包含所有风险加载pipeline

    返回
    ----
    pipeline:Pipeline
        包含风险模型中每个因子的风险加载的pipeline
    """
    return Pipeline(
        columns={
            'BasicMaterials': BasicMaterials(),
            'ConsumerCyclical': ConsumerCyclical(),
            'FinancialServices': FinancialServices(),
            'RealEstate': RealEstate(),
            'ConsumerDefensive': ConsumerDefensive(),
            'HealthCare': HealthCare(),
            'Utilities': Utilities(),
            'CommunicationServices': CommunicationServices(),
            'Energy': Energy(),
            'Industrials': Industrials(),
            'Technology': Technology(),
            'Momentum': Momentum(),
            'Value': Value(),
            'Size': Size(),
            'ShortTermReversal': ShortTermReversal(),
            'Volatility': Volatility(),
        })
