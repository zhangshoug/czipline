"""
辅助函数
"""
from cswd.common.utils import ensure_list

from .core import symbols, to_tdates


def select_output_by(output, start=None, end=None, assets=None):
    """
    按时间及代码选择`pipeline`输出数据框
    
    专用于研究环境下的run_pipeline输出结果分析

    参数
    ----
    output : MultiIndex DataFrame
        pipeline输出结果
    start ： str
        开始时间
    end ： str
        结束时间    
    assets ： 可迭代对象或str
        股票代码

    案例
    ----  
    >>> # result 为运行`pipeline`输出结果 
    >>> select_output_by(result,'2018-04-23','2018-04-24',stock_codes=['000585','600871'])

                                                  mean_10
    2018-04-23 00:00:00+00:00 	*ST东电(000585) 	2.7900
                                *ST油服(600871) 	2.0316
    2018-04-24 00:00:00+00:00 	*ST东电(000585) 	2.7620
                                *ST油服(600871) 	2.0316    
    """
    nlevels = output.index.nlevels
    _, start, end = to_tdates(start, end)
    if nlevels != 2:
        raise ValueError('输入数据框只能是run_pipeline输出结果，MultiIndex DataFrame')
    if start:
        output = output.loc[start:]
    if end:
        output = output.loc[:end]
    if assets is not None:
        assets = symbols(assets)
        return output.loc[(slice(None), assets), :]
    else:
        return output