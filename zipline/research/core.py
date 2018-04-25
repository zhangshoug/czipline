"""
用途：
    1. notebook研究
    2. 测试
"""
import pandas as pd

from cswd.common.utils import sanitize_dates

from zipline.pipeline import Pipeline
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.data import CNEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.fundamentals import Fundamentals
from zipline.data.bundles.core import load

from zipline.pipeline.loaders.blaze import global_loader

bundle = 'cndaily'

bundle_data = load(bundle)

pipeline_loader = USEquityPricingLoader(bundle_data.equity_daily_bar_reader,
                                        bundle_data.adjustment_reader,)


def choose_loader(column):
    if column in CNEquityPricing.columns:
        return pipeline_loader
    elif Fundamentals.has_column(column):
        return global_loader
    raise ValueError("No PipelineLoader registered for column %s." % column)


def run_pipeline(pipe, start, end):
    calendar = bundle_data.equity_daily_bar_reader.trading_calendar
    dates = calendar.all_sessions
    # 修正日期
    start, end = sanitize_dates(start, end)
    # 定位交易日期
    start_date = dates[dates.get_loc(start, method='bfill')]
    end_date = dates[dates.get_loc(end, method='ffill')]
    finder = bundle_data.asset_finder
    df = SimplePipelineEngine(
        choose_loader,
        dates,
        finder,
    ).run_pipeline(
        pipe, start_date, end_date
    )
    return df