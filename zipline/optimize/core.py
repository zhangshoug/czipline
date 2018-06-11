import logbook
import numpy as np
import pandas as pd
from toolz import concat
import cvxpy as cvx

from .result import (InfeasibleConstraints, OptimizationFailed,
                     OptimizationResult, UnboundedObjective)

logger = logbook.Logger('投资组合优化')

__all__ = ['calculate_optimal_portfolio', 'run_optimization']


def _run(objective, constraints, current_weights):
    if current_weights is not None:
        if len(current_weights) == 0:
            current_weights = None
    cvx_objective = objective.to_cvxpy(current_weights)
    new_weights = objective.new_weights
    new_weights_series = objective.new_weights_series
    constraint_map = {
        c: c.to_cvxpy(new_weights, new_weights_series, current_weights)
        for c in constraints
    }
    cvx_constraints = list(concat(constraint_map.values()))

    problem = cvx.Problem(cvx_objective, cvx_constraints)

    problem.solve()
    return problem, cvx_objective, constraint_map


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
    assert isinstance(constraints, list), 'constraints应该为列表类型'
    problem, cvx_objective, constraint_map = _run(objective, constraints,
                                                  current_portfolio)
    result = OptimizationResult(problem, objective, current_portfolio,
                                cvx_objective, constraint_map)
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
        raise UnboundedObjective('问题无界。如没有限制总权重，求解最大alpha时，目标权重无上界')
    elif status == 'infeasible':
        info = result.diagnostics
        raise InfeasibleConstraints(info)
    elif status == 'optimal':
        return result.new_weights