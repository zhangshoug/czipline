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
from .fundamentals import Fundamentals
from .pipeline import Pipeline
from .mixins import SingleInputMixin

__all__ = ['Momentum', 'Value', 'Size', 'ShortTermReversal', 'Volatility']

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


class BasicMaterials(CustomFactor):
    """
    Risk Model loadings for the basic materials sector.
    """
    sector = Fundamentals.info.sector_code.latest


class Momentum(CustomFactor):
    """
    Risk Model loadings for the “momentum” style factor.

    This factor captures differences in returns between stocks that 
    have had large gains in the last 11 months and stocks that have 
    had large losses in the last 11 months.
    """
    inputs = [USEquityPricing.close]
    window_length = 12 * PPM

    def compute(self, today, assets, out, closes):
        part = closes[:11 * PPM + 1]
        ratio = (np.diff(part, axis=0) / part[1:])
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
        r = (np.diff(closes, axis=0) / closes[1:]) - 1
        out[:] = zscore(r.std(axis=0))


def risk_loading_pipeline():
    """
    为风险模型创建一个包含所有风险加载pipeline

    返回
    ----
    pipeline:Pipeline
        包含风险模型中每个因子的风险加载的pipeline
    """
    pass
