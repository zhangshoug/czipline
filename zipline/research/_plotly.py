"""
简单版本量化分析图

定义标准ohlcv技术指标图，简化plotly配置

无论是网页，还是Jupyter Notebook，光标悬停位置都可以读取日期信息，所以不显示x轴刻度
而各项目名称包含了指标名称，故无需再使用图例

建议使用`cufflinks`
"""
import pytz
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly import tools

from cswd.common.utils import ensure_list

from ._talib import indicators
from .core import ohlcv, symbols

ALLOWED = ('RSI', 'MACD', 'CCI', 'ROC', 'ROCR100')


def _fig_default_setting():
    """技术图默认设置"""
    fig_setting = {}
    fig_setting['volume'] = {'ns': (5, 10, 20), 'field': 'volume'}
    fig_setting['rsi'] = {'ns': (6, 12, 24)}
    return fig_setting


def _get_date_hovertext(data):
    hovertext = []
    for d in data.index:
        hovertext.append('date: ' + d.strftime(r'%Y-%m-%d'))
    return hovertext


def _Candlestick(kwargs):
    """股价蜡烛图"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    data = df.loc[limit_start:limit_end, :]
    hovertext = _get_date_hovertext(data)
    trace = go.Candlestick(
        x=np.arange(data.shape[0]),
        open=data.open,
        high=data.high,
        low=data.low,
        close=data.close,
        text=hovertext,
        increasing={'line': {
            'color': 'red'
        }},
        decreasing={'line': {
            'color': 'green'
        }},
        name='price')
    return [trace]


def _Volume(kwargs):
    """成交量图"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    colors = kwargs.get('colors')
    data = df.loc[limit_start:limit_end, :]
    hovertext = _get_date_hovertext(data)
    trace = go.Bar(
        x=np.arange(data.shape[0]),
        y=data.volume,
        text=hovertext,
        marker=dict(color=colors),
        name='volume')
    return [trace]


def _MACD(kwargs):
    """绘制MACD"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    colors = kwargs.get('colors')
    inds = indicators('MACD', df).loc[limit_start:limit_end, :]
    hovertext = _get_date_hovertext(inds)
    x = np.arange(inds.shape[0])
    macd = go.Scatter(x=x, y=inds['macd'], name='macd_macd')
    signal = go.Scatter(
        x=x,
        y=inds['macdsignal'],
        name='macd_signal',
        line=dict(dash='dot'),
        text=hovertext,
    )
    hist = go.Bar(
        x=x,
        y=inds['macdhist'],
        width=[0.01] * inds.shape[0],
        marker=dict(color=colors),
        name='macd_hist')
    return [macd, signal, hist]


def _RSI(kwargs):
    """绘制RSI"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    v_setting = kwargs.get('rsi')
    ns = v_setting.get('ns')
    traces = []
    for i, n in enumerate(ns):
        s = indicators('rsi', df, timeperiod=n).loc[limit_start:limit_end]
        name = 'rsi_{}'.format(n)
        if i == 0:
            hovertext = _get_date_hovertext(s)
            trace = go.Scatter(
                x=np.arange(len(s)),
                y=s.values,
                text=hovertext,
                mode='lines',
                name=name)
        else:
            trace = go.Scatter(
                x=np.arange(len(s)), y=s.values, mode='lines', name=name)
        traces.append(trace)
    return traces


def _CCI(kwargs):
    """绘制CCI"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    traces = []
    n = 14
    s = indicators('cci', df, timeperiod=n).loc[limit_start:limit_end]
    hovertext = _get_date_hovertext(s)
    name = 'cci_{}'.format(n)
    trace = go.Scatter(
        x=np.arange(len(s)),
        y=s.values,
        text=hovertext,
        mode='lines',
        name=name)
    traces.append(trace)
    return traces


def _ROC(kwargs):
    """绘制ROC"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    traces = []
    n = 10
    s = indicators('roc', df, timeperiod=n).loc[limit_start:limit_end]
    hovertext = _get_date_hovertext(s)
    name = 'roc_{}_days'.format(n)
    trace = go.Scatter(
        x=np.arange(len(s)),
        y=s.values,
        text=hovertext,
        mode='lines',
        name=name)
    traces.append(trace)
    return traces


def _ROCR100(kwargs):
    """绘制ROC"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    traces = []
    n = 10
    s = indicators('rocr100', df, timeperiod=n).loc[limit_start:limit_end]
    hovertext = _get_date_hovertext(s)
    name = 'rocr100_{}_days'.format(n)
    trace = go.Scatter(
        x=np.arange(len(s)),
        y=s.values,
        text=hovertext,
        mode='lines',
        name=name)
    traces.append(trace)
    return traces


def _BBANDS(kwargs):
    """
    布林带

    技术参数
    -------
    使用21天，2倍
    """
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    ndays = 21
    inds = indicators(
        'BBANDS', df, timeperiod=ndays).loc[limit_start:limit_end, :]
    traces = []
    for c in inds.columns:
        name = 'price_{}_{}'.format(c, ndays)
        trace = go.Scatter(
            x=np.arange(inds.shape[0]),
            y=inds[c],
            name=name,
        )
        traces.append(trace)
    return traces


def _SMA(kwargs):
    """成交量简单移动平均"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    v_setting = kwargs.get('volume')
    ns = v_setting.get('ns')
    field = v_setting.get('field')
    traces = []
    for n in ns:
        s = indicators(
            'sma', df, timeperiod=n, price=field).loc[limit_start:limit_end]
        name = '{}_sma{}'.format(field, n)
        trace = go.Scatter(
            x=np.arange(len(s)), y=s.values, mode='lines', name=name)
        traces.append(trace)
    return traces


def _indicators_factory(ind, kwargs):
    """指标工厂"""
    ind = ind.upper()
    if ind == 'MACD':
        return _MACD(kwargs)
    elif ind == 'RSI':
        return _RSI(kwargs)
    elif ind == 'CCI':
        return _CCI(kwargs)
    elif ind == 'ROC':
        return _ROC(kwargs)
    elif ind == 'ROCR100':
        return _ROCR100(kwargs)


def _relayout(traces):
    """
    技术分析图布局

    布局说明
    1. 相对于其他附图(技术分析部分)，主图所占行数为附图三倍
    2. 理论上可以接受任意数量的trace，但是，超过3个不美观
    """
    row_dict = {}
    for trace in traces:
        name = trace.name
        row_key = name.split('_')[0]
        if row_key == 'price':
            # 主图放第一区(3行)
            row_dict[row_key] = 1
        elif row_key == 'volume':
            # 主图放第二区(1行)
            row_dict[row_key] = 4
        else:
            # 其他指标自第5行开始
            row = row_dict.get(row_key, 0)
            if not row:
                row_dict[row_key] = max(row_dict.values()) + 1
    rows = max(row_dict.values())
    specs = [[{'rowspan': 3, 'b': 0.02}]] + [[{'b': 0.02}]] * (rows - 1)
    fig = tools.make_subplots(
        rows=rows,
        specs=specs,
        shared_xaxes=True,
        start_cell='top-left',
        vertical_spacing=0.001)
    for trace in traces:
        name = trace.name
        row_key = name.split('_')[0]
        fig.append_trace(trace, row_dict[row_key], 1)
    return fig


def _gen_fig(asset, inds, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    assert end > start, '结束日期必须大于开始日期'
    inds = ensure_list(inds)
    asset = symbols([asset])[0]
    asset_name = asset.asset_name
    # 前移半年
    d_start = pd.Timestamp(start) - pd.Timedelta(days=180)
    df = ohlcv(asset, d_start, end)
    limit_start = df.index[df.index >= start][0]
    limit_end = df.index[df.index <= end][-1]
    # 颜色
    # colors = ['red' if x >= 0 else 'green' for x in df.close.pct_change()]
    diff = (df.close - df.open).loc[limit_start:limit_end]
    colors = ['red' if x >= 0 else 'green' for x in diff]
    title = '{}技术分析图({}到{})'.format(asset_name,
                                    limit_start.strftime(r'%Y-%m-%d'),
                                    limit_end.strftime(r'%Y-%m-%d'))
    fig_setting = _fig_default_setting()
    kwargs = {
        'df': df,
        'limit_start': limit_start,
        'limit_end': limit_end,
        'colors': colors,
    }
    kwargs.update(fig_setting)
    traces = []
    # 主图
    traces.extend(_Candlestick(kwargs))
    traces.extend(_BBANDS(kwargs))
    # 成交量图
    traces.extend(_Volume(kwargs))
    traces.extend(_SMA(kwargs))
    # 其他技术图
    for ind in inds:
        if ind.upper() not in ALLOWED:
            raise NotImplementedError('不支持指标{}\n，可接受的指标：{}'.format(
                ind, ALLOWED))
        traces.extend(_indicators_factory(ind, kwargs))
    fig = _relayout(traces)
    # 不使用Rangeslider及不显示x刻度
    layout = dict(
        title=title,
        showlegend=False,
        xaxis=dict(rangeslider=dict(visible=False), showticklabels=False))
    fig['layout'].update(layout)
    return fig


def plot_ohlcv(asset,
               inds,
               start=pd.datetime.today() - pd.Timedelta(days=60),
               end=pd.datetime.today()):
    """
    绘制python环境下的股票技术分析图

    参数
    ----
    asset:单个股票代码或资产对象
        要分析的股票
    inds:列表
        指标名称列表
    start:类似日期对象
        开始日期。默认当前日期前二个月
    end:类似日期对象
        结束日期。默认为当天

    用法
    ---
    >>> # 网页
    >>> plot_ohlcv('000333',['MACD','CCI'])
    >>> 
    """
    fig = _gen_fig(asset, inds, start, end)
    plotly.offline.plot(fig)


def iplot_ohlcv(asset,
                inds,
                start=pd.datetime.today() - pd.Timedelta(days=60),
                end=pd.datetime.today()):
    """
    绘制ipython环境下的股票技术分析图

    参数
    ----
    asset:单个股票代码或资产对象
        要分析的股票
    inds:列表
        指标名称列表
    start:类似日期对象
        开始日期。默认当前日期前二个月
    end:类似日期对象
        结束日期。默认为当天

    用法
    ---
    >>> # jupyter notebook
    >>> iplot_ohlcv('000333',['MACD','CCI'])
    >>> 
    """
    plotly.offline.init_notebook_mode(connected=True)
    fig = _gen_fig(asset, inds, start, end)
    plotly.offline.iplot(fig)


class AnalysisFigure(object):
    """技术分析图形类"""

    def __init__(self,
                 asset,
                 inds,
                 start=pd.datetime.today() - pd.Timedelta(days=60),
                 end=pd.datetime.today()):
        self._asset = asset
        self._inds = ensure_list(inds)
        self._start = start
        self._end = end
        self._fig = None

    def iplot(self):
        plotly.offline.init_notebook_mode(connected=True)
        if self._fig is None:
            self._fig = _gen_fig(self._asset, self._inds, self._start,
                                 self._end)
        plotly.offline.iplot(self._fig)

    def plot(self):
        if self._fig is None:
            self._fig = _gen_fig(self._asset, self._inds, self._start,
                                 self._end)
        plotly.offline.plot(self._fig)

    def indicator_setting(self, kwargs):
        """调整技术指标计算参数"""
        raise NotImplementedError('尚未完成')

    def change_asset(self, asset):
        """更改要绘制图形的股票"""
        self._asset = asset

    def add_ind(self, ind_name):
        """添加单个指标"""
        self._inds.append(ind_name)

    def remove_ind(self, ind_name):
        """移除单个指标"""
        self._inds.remove(ind_name)

    def add_signal(self, signals):
        """
        添加买入卖出信号(固定添加到第一区主图)
        在技术分析图上标注策略，便于直观分析策略的优劣

        signals:Pd.Series
            时间Index单列，数据只能为0，-1，1
        """
        msg = 'signals为时间序列，且值只能为"0，-1，1"三者之一'
        assert isinstance(signals, pd.Series), msg
        vs = set(signals.unique())
        if len(vs.difference((0, -1, 1))):
            raise ValueError(msg)
        if self._fig is None:
            self._fig = _gen_fig(self._asset, self._inds, self._start,
                                 self._end)
        if signals.index.tz != pytz.UTC:
            signals.index = signals.index.tz_localize('UTC').normalize()
        buy_signals = signals[signals >= 1]
        sell_signals = signals[signals <= -1]
        high = self._fig.data[0]['high']
        low = self._fig.data[0]['low']
        buy = go.Scatter(
            name='买入信号',
            mode='markers',
            x=[low.index.get_loc(x) for x in buy_signals.index],
            y=low.loc[buy_signals.index].values * 0.99,
            marker=dict(symbol="triangle-up", color='red'))
        sell = go.Scatter(
            name='卖出信号',
            mode='markers',
            x=[high.index.get_loc(x) for x in sell_signals.index],
            y=high.loc[sell_signals.index].values * 1.01,
            marker=dict(symbol="triangle-down", color='green'))
        self._fig.append_trace(buy, 1, 1)
        self._fig.append_trace(sell, 1, 1)
