
![Zipline](https://media.quantopian.com/logos/open_source/zipline-logo-03_.png)

Zipline是QUANTOPIAN开发的算法交易库。这是一个事件驱动，支持回测和实时交易的系统。
Zipline目前有一个免费的[回测平台](https://www.quantopian.com)，可以很方便编写交易策略和执行回测。为处理A股数据，增加或修改基础数据相关部分，用于本机策略回测分析。

## 功能

### 新增`USEquityPricing`数据列

+ `prev_close` 前收盘
+ `turnover` 换手率
+ `amount` 成交额
+ `tmv` 总市值
+ `cmv` 流通市值
+ `circulating_share` 流通股本
+ `total_share` 总股本

**注意** 以上数据不会因股票分红派息而调整

### 新增`Fundamentals`数据容器

`Fundamentals`是用于`pipeline`的数据集容器，包括偶发性变化及固定信息数据，如上市日期、行业分类、所属概念、财务数据等信息。

### 提供后台自动提取、转换数据脚本

通过定时后台计划任务，自动提取回测所需网络数据，并转换为符合`zipline`需求的数据格式，可以更加专注于编写策略及相关分析。

## 安装

### 克隆
$ git clone https://github.com/liudengfeng/czipline.git
### 安装包
# 转移至项目安装文件所在目录
$ pip install -r ./etc/requirements.txt
$ pip install -r ./etc/requirements_add.txt

# 下载并安装ta-lib包
### 编译`C`扩展库
$ python setup.py build_ext --inplace
### 安装`zipline`
$ python setup.py install
$ # 如需开发安装
$ python setup.py develop
## 准备数据

成功安装`zipline`后，`cswd`包也已经成功安装。在相应环境下，执行以下命令：

>$ init-stock-data # 初始化基础数据。首次耗时大约4小时，以后每日后台自动刷新约半小时

>$ zipline ingest -b cndaily # 转换日线数据，耗时约10分钟

>$ sql-to-bcolz # `Fundamentals`数据，耗时约1.5分钟

## 使用

计算`Fama-French`三因子案例涉及到：
1. 如何在`Notebook`运行回测
2. 选择基准收益率指数代码
3. 计划函数用法
4. `Fundamentals`及财务数据
5. `pipeline`及自定义因子用法
6. 回测速度

比较适合作为演示材料


```python
%load_ext zipline
```


```python
%%zipline --start 2017-1-1 --end 2018-1-1 --bm-symbol 399001
from zipline.api import symbol, sid, get_datetime

import pandas as pd
import numpy as np
from zipline.api import (attach_pipeline, pipeline_output, get_datetime,
                         calendars, schedule_function, date_rules)
from zipline.pipeline import Pipeline
from zipline.pipeline import CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.fundamentals import Fundamentals

# time frame on which we want to compute Fama-French
normal_days = 31
# approximate the number of trading days in that period
# this is the number of trading days we'll look back on,
# on every trading day.
business_days = int(0.69 * normal_days)


# 以下自定义因子选取期初数
class Returns(CustomFactor):
    """
    每个交易日每个股票窗口长度"business_days"期间收益率
    """
    window_length = business_days
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, price):
        out[:] = (price[-1] - price[0]) / price[0] * 100


class MarketEquity(CustomFactor):
    """
    每个交易日每只股票所对应的总市值
    """
    window_length = business_days
    inputs = [USEquityPricing.tmv]

    def compute(self, today, assets, out, mcap):
        out[:] = mcap[0]


class BookEquity(CustomFactor):
    """
    每个交易日每只股票所对应的账面价值（所有者权益）
    """
    window_length = business_days
    inputs = [Fundamentals.balance_sheet.A107]

    def compute(self, today, assets, out, book):
        out[:] = book[0]


def initialize(context):
    """
    use our factors to add our pipes and screens.
    """
    pipe = Pipeline()
    mkt_cap = MarketEquity()
    pipe.add(mkt_cap, 'market_cap')

    book_equity = BookEquity()
    # book equity over market equity
    be_me = book_equity / mkt_cap
    pipe.add(be_me, 'be_me')

    returns = Returns()
    pipe.add(returns, 'returns')

    attach_pipeline(pipe, 'ff_example')
    schedule_function(
        func=myfunc,
        date_rule=date_rules.month_end())


def before_trading_start(context, data):
    """
    every trading day, we use our pipes to construct the Fama-French
    portfolios, and then calculate the Fama-French factors appropriately.
    """

    factors = pipeline_output('ff_example')

    # get the data we're going to use
    returns = factors['returns']
    mkt_cap = factors.sort_values(['market_cap'], ascending=True)
    be_me = factors.sort_values(['be_me'], ascending=True)

    # to compose the six portfolios, split our universe into portions
    half = int(len(mkt_cap) * 0.5)
    small_caps = mkt_cap[:half]
    big_caps = mkt_cap[half:]

    thirty = int(len(be_me) * 0.3)
    seventy = int(len(be_me) * 0.7)
    growth = be_me[:thirty]
    neutral = be_me[thirty:seventy]
    value = be_me[seventy:]

    # now use the portions to construct the portfolios.
    # note: these portfolios are just lists (indices) of equities
    small_value = small_caps.index.intersection(value.index)
    small_neutral = small_caps.index.intersection(neutral.index)
    small_growth = small_caps.index.intersection(growth.index)

    big_value = big_caps.index.intersection(value.index)
    big_neutral = big_caps.index.intersection(neutral.index)
    big_growth = big_caps.index.intersection(growth.index)

    # take the mean to get the portfolio return, assuming uniform
    # allocation to its constituent equities.
    sv = returns[small_value].mean()
    sn = returns[small_neutral].mean()
    sg = returns[small_growth].mean()

    bv = returns[big_value].mean()
    bn = returns[big_neutral].mean()
    bg = returns[big_growth].mean()

    # computing SMB
    context.smb = (sv + sn + sg) / 3 - (bv + bn + bg) / 3

    # computing HML
    context.hml = (sv + bv) / 2 - (sg + bg) / 2


def myfunc(context, data):
    d = get_datetime('Asia/Shanghai')
    print(d, context.smb, context.hml)
```

    2017-01-26 15:00:00+08:00 0.014014289806335789 6.605843892342312
    2017-02-28 15:00:00+08:00 4.1169182374497195 7.690119769984805
    2017-03-31 15:00:00+08:00 0.35808304923773615 2.7492806758694215
    2017-04-28 15:00:00+08:00 -4.318408584890385 5.414312699826368
    2017-05-31 15:00:00+08:00 -0.4828317045367072 3.0869028143557147
    2017-06-30 15:00:00+08:00 0.8640245866550513 0.09803178533289003
    2017-07-31 15:00:00+08:00 -2.3024594948720227 6.2829537294457145
    2017-08-31 15:00:00+08:00 3.2003154621799155 2.269609384481118
    2017-09-29 15:00:00+08:00 1.1669055941862554 -0.6079568594636064
    2017-10-31 15:00:00+08:00 -1.6233534895267374 -0.795885505339075
    2017-11-30 15:00:00+08:00 -2.965097825507776 4.4434701009908615
    2017-12-29 15:00:00+08:00 -1.1942883365086068 -0.38062423581176485
    [2018-05-02 00:57:22.446560] INFO: zipline.finance.metrics.tracker: Simulated 244 trading days
    first open: 2017-01-03 01:31:00+00:00
    last close: 2017-12-29 07:00:00+00:00





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>algo_volatility</th>
      <th>algorithm_period_return</th>
      <th>alpha</th>
      <th>benchmark_period_return</th>
      <th>benchmark_volatility</th>
      <th>beta</th>
      <th>capital_used</th>
      <th>ending_cash</th>
      <th>ending_exposure</th>
      <th>ending_value</th>
      <th>...</th>
      <th>short_exposure</th>
      <th>short_value</th>
      <th>shorts_count</th>
      <th>sortino</th>
      <th>starting_cash</th>
      <th>starting_exposure</th>
      <th>starting_value</th>
      <th>trading_days</th>
      <th>transactions</th>
      <th>treasury_period_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-03 07:00:00+00:00</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.008422</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-04 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.020412</td>
      <td>0.038928</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-05 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.019096</td>
      <td>0.108456</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-06 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.011028</td>
      <td>0.143696</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-09 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015197</td>
      <td>0.124811</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-10 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.012696</td>
      <td>0.117206</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-11 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.003768</td>
      <td>0.125535</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-12 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.004510</td>
      <td>0.126307</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-13 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.016590</td>
      <td>0.133134</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-16 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.045626</td>
      <td>0.187327</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-17 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.034425</td>
      <td>0.194146</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-18 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.036590</td>
      <td>0.185154</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-19 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.040145</td>
      <td>0.177295</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-20 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.026628</td>
      <td>0.185263</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-23 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.019745</td>
      <td>0.182241</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-24 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.023149</td>
      <td>0.176279</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-25 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.019571</td>
      <td>0.171793</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-01-26 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.012291</td>
      <td>0.169693</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-03 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.016930</td>
      <td>0.165575</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-06 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.009670</td>
      <td>0.163786</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-07 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.011946</td>
      <td>0.159770</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-08 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.004621</td>
      <td>0.158217</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-09 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000548</td>
      <td>0.155592</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-10 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000906</td>
      <td>0.152175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-13 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.009204</td>
      <td>0.151233</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-14 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.008624</td>
      <td>0.148209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-15 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000010</td>
      <td>0.147860</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-16 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.007547</td>
      <td>0.146827</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-17 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.002041</td>
      <td>0.145183</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-02-20 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014921</td>
      <td>0.147361</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-11-20 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.123849</td>
      <td>0.132024</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>215</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-21 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.140018</td>
      <td>0.132558</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>216</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-22 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.135959</td>
      <td>0.132329</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>217</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-23 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098098</td>
      <td>0.136977</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>218</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-24 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.097403</td>
      <td>0.136668</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>219</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-27 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076355</td>
      <td>0.137966</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>220</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-28 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.089812</td>
      <td>0.138260</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>221</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-29 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.088927</td>
      <td>0.137953</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>222</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-11-30 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.075365</td>
      <td>0.138321</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>223</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-01 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.082150</td>
      <td>0.138155</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>224</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-04 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.082288</td>
      <td>0.137846</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>225</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-05 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.066587</td>
      <td>0.138436</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>226</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-06 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.072146</td>
      <td>0.138226</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>227</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-07 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.061330</td>
      <td>0.138356</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>228</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-08 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.074478</td>
      <td>0.138634</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>229</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-11 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.094936</td>
      <td>0.139707</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>230</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-12 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.085106</td>
      <td>0.139749</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>231</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-13 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.094999</td>
      <td>0.139743</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>232</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-14 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.091686</td>
      <td>0.139487</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>233</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-15 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.080676</td>
      <td>0.139614</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>234</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-18 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.076942</td>
      <td>0.139371</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>235</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-19 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.088230</td>
      <td>0.139468</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>236</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-20 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081188</td>
      <td>0.139352</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>237</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-21 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.092479</td>
      <td>0.139444</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>238</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-22 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.090112</td>
      <td>0.139175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>239</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-25 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081401</td>
      <td>0.139150</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>240</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-26 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.083006</td>
      <td>0.138864</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>241</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-27 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.072136</td>
      <td>0.138982</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>242</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-28 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.078307</td>
      <td>0.138805</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>243</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2017-12-29 07:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.084836</td>
      <td>0.138640</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>None</td>
      <td>10000000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>244</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 37 columns</p>
</div>



**运行时长10-12秒**

+ 有关如何使用，请参考[quantopian使用手册](https://www.quantopian.com/help)。
+ 有关本项目的说明，请参阅[介绍材料](https://github.com/liudengfeng/czipline/tree/master/docs/%E4%BB%8B%E7%BB%8D%E6%9D%90%E6%96%99)
+ 有关后台自动数据处理，请参考[脚本](https://github.com/liudengfeng/czipline/blob/master/docs/%E4%BB%8B%E7%BB%8D%E6%9D%90%E6%96%99/bg_zipline_tasks.cron)

**特别说明**：个人当前使用Ubuntu17.10操作系统

## 参考配置

![系统](./sys.png)

+ Ubuntu 17.10
+ Anaconda 
+ python 3.6

## 交流

该项目纯属个人爱好，水平有限，欢迎加入来一起完善。

**添加个人微信，请务必备注`zipline`**


```python
%%html
<img src="./images/ldf.png",width=40px,height=40px>
```


<img src="./images/ldf.png",width=40px,height=40px>

