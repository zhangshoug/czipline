#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note that part of the API is implemented in TradingAlgorithm as
# methods (e.g. order). These are added to this namespace via the
# decorator ``api_method`` inside of algorithm.py.
from .finance.asset_restrictions import (
    Restriction,
    StaticRestrictions,
    HistoricalRestrictions,
    RESTRICTION_STATES,
)
from .finance import commission, execution, slippage, cancel_policy
from .finance.cancel_policy import (
    NeverCancel,
    EODCancel
)
from .finance.slippage import (
    FixedSlippage,
    FixedBasisPointsSlippage,
    VolumeShareSlippage,
)
from .utils import math_utils, events
from .utils.events import (
    calendars,
    date_rules,
    time_rules
)

import numpy as np
import pandas as pd


def last_trade(context, data, sid):
    """
    给定股票最近交易价格

    parameters
    ----------

    context:
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    sid : zipline Asset
        the sid price

    """
    try:
        # # 签名更改
        return data.current(sid, 'price')
    except KeyError:
        return context.portfolio.positions[sid].last_sale_price



def get_current_holdings(context, data):
    """
    当前组合头寸。即股票持有的数量。

    parameters
    ----------

    context :
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    returns
    -------
    pandas.Series
        contracts held of each asset

    """
    positions = context.portfolio.positions
    return pd.Series({stock: pos.amount
                      for stock, pos in positions.items()})



def get_current_allocations(context, data):
    """
    当前资产组合中各股票的权重

    w = 持有股票市值 / 资产组合市值

    parameters
    ----------

    context :
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    returns
    -------
    pandas.Series
        current allocations as a percent of total liquidity

    notes
    -----
    资产组合市值 = 持有股票市值 + 现金

    """
    holdings = get_current_holdings(context, data)
    prices = pd.Series({sid: last_trade(context, data, sid)
                        for sid in holdings.index})
    return prices * holdings / context.portfolio.portfolio_value


__all__ = [
    'EODCancel',
    'FixedSlippage',
    'FixedBasisPointsSlippage',
    'NeverCancel',
    'VolumeShareSlippage',
    'Restriction',
    'StaticRestrictions',
    'HistoricalRestrictions',
    'RESTRICTION_STATES',
    'cancel_policy',
    'commission',
    'date_rules',
    'events',
    'execution',
    'math_utils',
    'slippage',
    'time_rules',
    'calendars',
    'get_current_holdings',
    'get_current_allocations'
]
