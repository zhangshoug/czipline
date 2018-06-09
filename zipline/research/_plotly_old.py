"""
定义标准ohlcv绘图，简化plotly配置
"""
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly import tools

from cswd.common.utils import ensure_list

from ._talib import indicators
from .core import ohlcv, symbols

ALLOWED = ('RSI', 'MACD', 'CCI')


def _fig_default_setting():
    """技术图默认设置"""
    fig_setting = {}
    fig_setting['volume'] = {'ns': (5, 10, 20), 'field': 'volume'}
    fig_setting['rsi'] = {'ns': (6, 12, 24)}
    return fig_setting


def _Candlestick(kwargs):
    """股价蜡烛图"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    hovertext = []
    data = df.loc[limit_start:limit_end, :]
    for d in data.index:
        hovertext.append('date:' + d.strftime(r'%Y-%m-%d'))
    trace = go.Candlestick(
        x=data.index,
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
    trace = go.Bar(
        x=data.index, y=data.volume, marker=dict(color=colors), name='volume')
    return [trace]


def _MACD(kwargs):
    """绘制MACD"""
    df = kwargs.get('df')
    limit_start = kwargs.get('limit_start')
    limit_end = kwargs.get('limit_end')
    colors = kwargs.get('colors')
    inds = indicators('MACD', df).loc[limit_start:limit_end, :]
    macd = go.Scatter(x=inds.index, y=inds['macd'], name='macd_macd')
    signal = go.Scatter(
        x=inds.index,
        y=inds['macdsignal'],
        name='macd_signal',
    )
    width = [100 * 1] * len(inds)
    hist = go.Bar(
        x=inds.index,
        y=inds['macdhist'],
        width=width,
        marker=dict(color=colors, line=dict(width=1.5)),
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
    for n in ns:
        s = indicators('rsi', df, timeperiod=n).loc[limit_start:limit_end]
        name = 'rsi_{}'.format(n)
        trace = go.Scatter(x=s.index, y=s.values, mode='lines', name=name)
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
    name = 'cci_{}'.format(n)
    trace = go.Scatter(x=s.index, y=s.values, mode='lines', name=name)
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
    inds = indicators(
        'BBANDS', df, timeperiod=21).loc[limit_start:limit_end, :]
    traces = []
    for c in inds.columns:
        name = 'price_{}'.format(c)
        trace = go.Scatter(
            x=inds.index,
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
        trace = go.Scatter(x=s.index, y=s.values, mode='lines', name=name)
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
    specs = [[{'rowspan': 3, 'b': 0.2}]] + [[{'b': 0.02}]] * (rows - 1)
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


def _fig(asset, inds, start, end):
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
        if ind not in ALLOWED:
            raise NotImplementedError('不支持指标{}\n，可接受的指标：{}'.format(
                ind, ALLOWED))
        traces.extend(_indicators_factory(ind, kwargs))
    fig = _relayout(traces)
    # 不使用Rangeslider
    layout = dict(
        title=title,
        showlegend=False,
        xaxis=dict(rangeslider=dict(visible=False)))
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
    >>> fig = ohlcv_fig('000333')
    >>> 
    """
    fig = _fig(asset, inds, start, end)
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
    >>> fig = ohlcv_fig('000333')
    >>> 
    """
    plotly.offline.init_notebook_mode(connected=True)
    fig = _fig(asset, inds, start, end)
    plotly.offline.iplot(fig)
