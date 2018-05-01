
.. figure:: https://media.quantopian.com/logos/open_source/zipline-logo-03_.png
   :alt: Zipline

   Zipline

Zipline是QUANTOPIAN开发的算法交易库。这是一个事件驱动，支持回测和实时交易的系统。
Zipline目前有一个免费的\ `回测平台 <https://www.quantopian.com>`__\ ，可以很方便编写交易策略和执行回测。为处理A股数据，增加或修改基础数据相关部分，用于本机策略回测分析。

功能
----

新增\ ``USEquityPricing``\ 数据列
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``prev_close`` 前收盘
-  ``turnover`` 换手率
-  ``amount`` 成交额
-  ``tmv`` 总市值
-  ``cmv`` 流通市值
-  ``circulating_share`` 流通股本
-  ``total_share`` 总股本

**注意** 以上数据不会因股票分红派息而调整

新增\ ``Fundamentals``\ 数据容器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Fundamentals``\ 是用于\ ``pipeline``\ 的数据集容器，包括偶发性变化及固定信息数据，如上市日期、行业分类、所属概念、财务数据等信息。

提供后台自动提取、转换数据脚本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过定时后台计划任务，自动提取回测所需网络数据，并转换为符合\ ``zipline``\ 需求的数据格式，可以更加专注于编写策略及相关分析。

安装
----

克隆
~~~~

$ git clone https://github.com/liudengfeng/czipline.git

安装包
~~~~~~

# 转移至项目安装文件所在目录
$ pip install -r ./etc/requirements.txt
$ pip install -r ./etc/requirements_add.txt

# 下载并安装ta-lib包

编译\ ``C``\ 扩展库
~~~~~~~~~~~~~~~~~~~

$ python setup.py build_ext --inplace

安装\ ``zipline``
~~~~~~~~~~~~~~~~~

$ python setup.py install
$ # 如需开发安装
$ python setup.py develop

准备数据
--------

成功安装\ ``zipline``\ 后，\ ``cswd``\ 包也已经成功安装。在相应环境下，执行以下命令：

    $ init-stock-data #
    初始化基础数据。首次耗时大约4小时，以后每日后台自动刷新约半小时

    $ zipline ingest -b cndaily # 转换日线数据，耗时约10分钟

    $ sql-to-bcolz # ``Fundamentals``\ 数据，耗时约1.5分钟

使用
----

计算\ ``Fama-French``\ 三因子案例涉及到： 1.
如何在\ ``Notebook``\ 运行回测 2. ``Fundamentals``\ 及财务数据 3.
``pipeline``\ 及自定义因子用法 4. 回测速度

比较适合作为演示材料

.. code:: ipython3

    %load_ext zipline

.. code:: ipython3

    %%zipline --start 2017-1-1 --end 2018-1-1
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


.. parsed-literal::

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
    [2018-05-02 00:17:39.152655] INFO: zipline.finance.metrics.tracker: Simulated 244 trading days
    first open: 2017-01-03 01:31:00+00:00
    last close: 2017-12-29 07:00:00+00:00




.. raw:: html

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
          <td>0.009712</td>
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
          <td>0.017593</td>
          <td>0.021406</td>
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
          <td>0.017435</td>
          <td>0.083084</td>
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
          <td>0.011356</td>
          <td>0.115404</td>
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
          <td>0.016261</td>
          <td>0.100950</td>
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
          <td>0.014560</td>
          <td>0.095760</td>
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
          <td>0.007377</td>
          <td>0.104382</td>
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
          <td>0.002279</td>
          <td>0.102578</td>
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
          <td>0.002971</td>
          <td>0.095975</td>
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
          <td>0.002830</td>
          <td>0.090519</td>
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
          <td>0.004917</td>
          <td>0.086298</td>
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
          <td>0.008848</td>
          <td>0.083788</td>
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
          <td>0.005804</td>
          <td>0.081915</td>
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
          <td>0.013538</td>
          <td>0.084470</td>
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
          <td>0.016315</td>
          <td>0.081719</td>
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
          <td>0.016426</td>
          <td>0.079044</td>
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
          <td>0.019886</td>
          <td>0.077078</td>
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
          <td>0.023528</td>
          <td>0.075314</td>
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
          <td>0.016438</td>
          <td>0.079092</td>
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
          <td>0.019071</td>
          <td>0.077224</td>
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
          <td>0.016799</td>
          <td>0.076073</td>
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
          <td>0.022117</td>
          <td>0.075736</td>
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
          <td>0.026046</td>
          <td>0.074588</td>
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
          <td>0.031241</td>
          <td>0.074054</td>
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
          <td>0.038126</td>
          <td>0.074482</td>
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
          <td>0.037984</td>
          <td>0.073157</td>
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
          <td>0.033727</td>
          <td>0.073710</td>
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
          <td>0.039533</td>
          <td>0.073515</td>
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
          <td>0.033644</td>
          <td>0.075131</td>
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
          <td>0.048734</td>
          <td>0.083479</td>
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
          <td>0.251884</td>
          <td>0.089501</td>
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
          <td>0.274200</td>
          <td>0.091110</td>
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
          <td>0.277181</td>
          <td>0.090908</td>
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
          <td>0.239366</td>
          <td>0.096537</td>
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
          <td>0.239911</td>
          <td>0.096317</td>
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
          <td>0.223520</td>
          <td>0.097295</td>
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
          <td>0.225295</td>
          <td>0.097075</td>
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
          <td>0.224669</td>
          <td>0.096867</td>
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
          <td>0.210273</td>
          <td>0.097585</td>
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
          <td>0.207867</td>
          <td>0.097414</td>
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
          <td>0.214127</td>
          <td>0.097303</td>
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
          <td>0.220566</td>
          <td>0.097199</td>
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
          <td>0.213209</td>
          <td>0.097258</td>
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
          <td>0.199686</td>
          <td>0.097863</td>
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
          <td>0.209450</td>
          <td>0.097949</td>
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
          <td>0.229425</td>
          <td>0.099101</td>
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
          <td>0.213267</td>
          <td>0.099970</td>
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
          <td>0.223562</td>
          <td>0.100070</td>
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
          <td>0.216329</td>
          <td>0.100104</td>
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
          <td>0.202645</td>
          <td>0.100677</td>
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
          <td>0.203985</td>
          <td>0.100462</td>
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
          <td>0.219102</td>
          <td>0.100980</td>
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
          <td>0.217641</td>
          <td>0.100788</td>
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
          <td>0.228927</td>
          <td>0.100948</td>
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
          <td>0.224924</td>
          <td>0.100825</td>
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
          <td>0.220979</td>
          <td>0.100701</td>
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
          <td>0.224630</td>
          <td>0.100515</td>
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
          <td>0.205774</td>
          <td>0.101669</td>
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
          <td>0.214140</td>
          <td>0.101652</td>
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
          <td>0.217752</td>
          <td>0.101466</td>
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



**运行时长8秒**

-  有关如何使用，请参考\ `quantopian使用手册 <https://www.quantopian.com/help>`__\ 。
-  有关本项目的说明，请参阅\ `介绍材料 <https://github.com/liudengfeng/czipline/tree/master/docs/%E4%BB%8B%E7%BB%8D%E6%9D%90%E6%96%99>`__
-  有关后台自动数据处理，请参考\ `脚本 <https://github.com/liudengfeng/czipline/blob/master/docs/%E4%BB%8B%E7%BB%8D%E6%9D%90%E6%96%99/bg_zipline_tasks.cron>`__

**特别说明**\ ：个人当前使用Ubuntu17.10操作系统

参考配置
--------

.. figure:: ./sys.png
   :alt: 系统

   系统

-  Ubuntu 17.10
-  Anaconda
-  python 3.6

交流
----

该项目纯属个人爱好，水平有限。欢迎加入来一起完善。
**添加个人微信，请务必备注\ ``zipline``**

.. figure:: https://github.com/liudengfeng/czipline/blob/master/docs/%E4%BB%8B%E7%BB%8D%E6%9D%90%E6%96%99/ldf.png
   :alt: 联系方式

   联系方式
