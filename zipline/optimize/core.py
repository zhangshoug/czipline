import cvxpy as cvx
import logbook
import numpy as np
import pandas as pd
from toolz import concat
from .result import (OptimizationResult, InfeasibleConstraints,
                     UnboundedObjective, OptimizationFailed)

logger = logbook.Logger('投资组合优化')

__all__ = [
    'order_optimal_portfolio', 'calculate_optimal_portfolio',
    'run_optimization'
]


def _check_assets(objective, constraints, current_portfolio):
    """
    检查资产索引一致性
    1. objective与current_portfolio
    2. objective与constraints

    只有限定条件对象属性：assets索引对象非空才比较

    要求目标资产索引对象要全部包含限定对象索引

    目的在于防止变量位置与限定输入参数位置不一致，导致优化结果异常
    """
    # 当前组合所含资产必须全部包含于目标函数资产内
    if current_portfolio is not None:
        diff = current_portfolio.index.difference(objective.assets)
        if len(diff):
            raise ValueError('当前组合资产{}并不在目标资产列表内'.format(diff))

    # 如果单个限定对象的资产不为空，则应包含于目标函数资产内
    for con in constraints:
        if con.constraint_assets is not None:
            diff = con.constraint_assets.difference(objective.assets)
            if len(diff):
                raise ValueError('限定：{}所含资产{}不在目标资产清单内'.format(
                    con.__class__.__name__, diff))


def _check_constraints(constraints):
    """检查限定条件"""
    pass


def _check_problem():
    """检查问题状态"""
    pass


# def _run(objective, constraints, current_portfolio):
#     """运行求解，返回问题对象"""
#     init_w_s = current_portfolio
#     l_w = objective.long_w
#     s_w = objective.short_w
#     l_w_s = objective.long_weights_series
#     s_w_s = objective.short_weights_series
#     constraint_list = []
#     for cons_obj in constraints:
#         constraint_list.extend([
#             con for con in cons_obj.gen_constraints(l_w, s_w, l_w_s, s_w_s,
#                                                     init_w_s)
#         ])
#     prob = cvx.Problem(objective.objective, constraint_list)
#     prob.solve()
#     return prob


def _run(objective, constraints, current_portfolio):
    """运行求解，返回问题对象"""
    l_w = objective.long_w
    s_w = objective.short_w
    l_w_s = objective.long_weights_series
    s_w_s = objective.short_weights_series
    constraint_map = {
        c: c.gen_constraints(l_w, s_w, l_w_s, s_w_s, current_portfolio)
        for c in constraints
    }
    cvx_constraints = list(concat(constraint_map.values()))
    prob = cvx.Problem(objective.objective, cvx_constraints)
    prob.solve()
    return prob


def run_optimization(objective, constraints, current_portfolio=None):
    """
    运行投资组合优化

    Parameters
    ----------
    objective :Objective
        将要最大化或最小化目标
    constraints ：list[Constraint])
        新投资组合必须满足的约束列表      
    current_portfolio：pd.Series, 可选
        包含当前投资组合权重的系列，以投资组合的清算价值的百分比表示。
        当从交易算法调用时，current_portfolio的默认值是算法的当前投资组合；
        当交互调用时，current_portfolio的默认值是一个空组合。

    Returns
    -------
    result：zipline.optimize.OptimizationResult
        包含有关优化结果信息的对象

    See also
    --------
    zipline.optimize.OptimizationResult
    zipline.optimize.calculate_optimal_portfolio()   
    """
    # 检查current_portfolio与目标所含assets一致性
    _check_assets(objective, constraints, current_portfolio)
    assert isinstance(constraints, list), 'constraints应该为列表类型'
    prob = _run(objective, constraints, current_portfolio)
    result = OptimizationResult(prob, objective, current_portfolio)
    return result


def calculate_optimal_portfolio(objective, constraints,
                                current_portfolio=None):
    """
    计算给定目标及限制的投资组合最优权重

    Parameters
    ----------
    objective :Objective
        将要最大化或最小化目标
    constraints ：list[Constraint])
        新投资组合必须满足的约束列表  
    current_portfolio：pd.Series, 可选
        包含当前投资组合权重的系列，以投资组合的清算价值的百分比表示。
        当从交易算法调用时，current_portfolio的默认值是算法的当前投资组合；
        当交互调用时，current_portfolio的默认值是一个空组合。

    Returns
    -------
    optimal_portfolio (pd.Series)
        当前返回的是pd.DataFrame(多头、空头、合计权重)

        包含最大化（或最小化）目标而不违反任何约束条件的投资组合权重的系列。权重应该与
        `current_portfolio`同样的方式来表达。

    Raises
    ------
    InfeasibleConstraints
        Raised when there is no possible portfolio that satisfies the received 
        constraints.
    UnboundedObjective
        Raised when the received constraints are not sufficient to put an upper 
        (or lower) bound on the calculated portfolio weights.

    Notes

    This function is a shorthand for calling run_optimization, checking for an error, 
    and extracting the result’s new_weights attribute.

    If an optimization problem is feasible, the following are equivalent:

    >>> # Using calculate_optimal_portfolio.
    >>> weights = calculate_optimal_portfolio(objective, constraints, portfolio)
    >>> # Using run_optimization.
    >>> result = run_optimization(objective, constraints, portfolio)
    >>> result.raise_for_status()  # Raises if the optimization failed.
    >>> weights = result.new_weights

    See also
    ---------
    zipline.optimize.run_optimization()
    """
    result = run_optimization(objective, constraints, current_portfolio)
    status = result.prob.status
    if status == 'unbounded':
        raise UnboundedObjective('目标无界')
    elif status == 'infeasible':
        raise InfeasibleConstraints('限制不可行')
    v = result.prob.value
    if np.isposinf(v) or np.isneginf(v):
        raise OptimizationFailed('求解失败')
    return result.new_weights

def order_optimal_portfolio(objective, constraints):
    """
    计算优化投资组合，并放置实现该组合所必需的订单

    Parameters
    ----------
        objective (Objective)
            The objective to be minimized/maximized by the new portfolio.
        constraints (list[Constraint])
            Constraints that must be respected by the new portfolio.

    Raises
    ------
        InfeasibleConstraints
            Raised when there is no possible portfolio that satisfies the 
            received constraints.
        UnboundedObjective
            Raised when the received constraints are not sufficient to put 
            an upper (or lower) bound on the calculated portfolio weights.

    Returns
    -------	
        order_ids (pd.Series[Asset -> str])
            The unique identifiers for the orders that were placed.    
    """
    pass
