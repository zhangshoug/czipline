import cvxpy as cvx
from .utils import (OptimizationResult, InfeasibleConstraints,
                    UnboundedObjective, OptimizationFailed)


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
    # 传入伪目标对象
    obj = objective.objective
    w = objective.w
    # 传入伪限制对象
    cons = []
    for con in constraints:
        cons.extend(con.make_constraints(w))
    prob = cvx.Problem(obj, cons)
    prob.solve()
    return OptimizationResult(prob, w, current_portfolio)
