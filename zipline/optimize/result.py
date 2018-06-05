"""
优化结果模型

注
--
    有限检查限制条件冲突
"""
import collections
import re
from itertools import combinations

import numpy as np
import pandas as pd

import cvxpy as cvx

NUM_SIGN_PATTERN = re.compile(r'(-)\d{1,}.\d{1,}')
VAR_SIGN_PATTERN = re.compile(r'(-)new_weights')
INVALID_PATTERN = re.compile(r'\((new_weights)')
NUM_PATTERN = re.compile(r'[-]?(\d{1,}.\d{1,})')
VAR_ID_PATTERN = re.compile(r'.*?\[(\d{1,})\]')


class InfeasibleConstraints(Exception):
    """

    Raised when an optimization fails because there are no valid portfolios.

    This most commonly happens when the weight in some asset is simultaneously 
    constrained to be above and below some threshold.
    """
    pass


class UnboundedObjective(Exception):
    """

    Raised when an optimization fails because at least one weight in the ‘optimal’ 
    portfolio is ‘infinity’.

    More formally, raised when an optimization fails because the value of an 
    objective function improves as a value being optimized grows toward infinity, 
    and no constraint puts a bound on the magnitude of that value.
    """
    pass


class OptimizationFailed(Exception):
    """
    Generic exception raised when an optimization fails a reason with no special 
    metadata.
    """
    pass


def search_v(expr, pat):
    s = re.search(pat, expr)
    if s is None:
        return None
    else:
        return s.groups()[0]


def parse_constraint_range(constraint):
    """解析单个限制其权重范围信息"""
    min_, max_ = -np.inf, np.inf
    expr = str(constraint)
    # 忽略
    invalid = search_v(expr, INVALID_PATTERN)
    if invalid:
        invalid = True
    else:
        invalid = False
    # 常量符号
    c_sign = search_v(expr, NUM_SIGN_PATTERN)
    if c_sign is None:
        c_sign = 1
    else:
        c_sign = -1
    # 变量符号
    v_sign = search_v(expr, VAR_SIGN_PATTERN)
    if v_sign is None:
        v_sign = 1
    else:
        v_sign = -1

    # 系数值
    c_num = search_v(expr, NUM_PATTERN)
    if c_num is not None:
        c_num = c_sign * float(c_num)

    # 变量切片序号
    id_num = search_v(expr, VAR_ID_PATTERN)
    if id_num is not None:
        id_num = int(id_num)

    if invalid:
        return (None, None, None)
    else:
        if isinstance(constraint, cvx.Zero):
            min_ = c_num * -1
            max_ = c_num * -1 #* c_sign #* v_sign
        elif isinstance(constraint, cvx.NonPos):
            if v_sign == -1:
                min_ = c_num
            else:
                max_ = c_num * -1
    return (id_num, min_, max_)


def gen_ranges(n, rngs):
    res = {k: [-np.inf, np.inf] for k in range(n)}
    for r in rngs:
        if r[0] is not None:
            # 更新下限
            old = res[r[0]][0]
            new = r[1]
            res[r[0]][0] = max(old, new)
            # 更新上限
            old = res[r[0]][1]
            new = r[2]
            res[r[0]][1] = min(old, new)
        else:
            for i in range(n):
                res[i][0] = -np.inf if r[1] is None else r[1]
                res[i][1] = np.inf if r[2] is None else r[2]
    return res


def check_violate(limit_a, limit_b):
    """如不相交，违背"""
    if (limit_a[0] > limit_b[1]) or (limit_b[0] > limit_a[1]):
        return True
    return False


def run_diagnostics(objective, new_weights, cvx_objective, constraint_map):
    info = '没有找到满足所有必需约束的投资组合，尝试优化失败。'
    info += '检查以下投资组合，发现违背约束条件：\n'
    asstes = objective.new_weights_series.index
    diagnostics = {}
    n = new_weights.size
    for k, cons in constraint_map.items():
        rngs = []
        for c in cons:
            rngs.append(parse_constraint_range(c))
        diagnostics[k] = gen_ranges(n, rngs)
    for k1, k2 in combinations(diagnostics.keys(), 2):
        limit_as = diagnostics[k1]
        limit_bs = diagnostics[k2]
        for i, (limit_a, limit_b) in enumerate(
                zip(limit_as.values(), limit_bs.values())):
            if check_violate(limit_a, limit_b):
                asset = asstes[i]
                msg = '{}不可能同时满足<{}>和<{}>约束\n'.format(asset, k1, k2)
                info += msg
    return info


class OptimizationResult():
    def __init__(self, prob, objective, old_weights, cvx_objective,
                 constraint_map):
        self.prob = prob
        self.objective = objective
        self._old_weights = old_weights
        self._cvx_objective = cvx_objective
        self._constraint_map = constraint_map

    def raise_for_status(self):
        """
        Raise an error if the optimization did not succeed.
        """
        if self.prob.status != 'optimal':
            raise NotImplementedError('未成功优化')

    def _diagnostics_str(self):
        prob = self.prob
        data = collections.OrderedDict({
            '最优解':
            prob.value,
            '迭代次数':
            prob.solver_stats.num_iters,
            '设立问题用时(s)':
            prob.solver_stats.setup_time,
            '求解问题用时(s)':
            prob.solver_stats.solve_time,
            '求解器名称':
            prob.solver_stats.solver_name,
        })
        return (pd.Series(data=data).to_string(float_format='{:,.6f}'.format))

    def print_diagnostics(self):
        """
        Print diagnostic information gathered during the optimization.
        """
        status = self.prob.status
        if status == 'unbounded':
            info = '问题无界。如没有限制总权重，求解最大alpha时，目标权重无上界'
        elif status == 'infeasible':
            info = self.diagnostics
        elif status == 'optimal':
            info = self._diagnostics_str()
        print(info)

    @property
    def old_weights(self):
        """
        pandas.Series 或 None
        Portfolio weights before the optimization.
        """
        return self._old_weights

    @property
    def new_weights(self):
        """
        新权重
        pandas.Series 或 None
            
        New optimal weights, or None if the optimization failed.

        """
        if not self.success:
            return None
        return self.objective.new_weights_value

    @property
    def diagnostics(self):
        """
        zipline.optimize.Diagnostics

        Object containing diagnostic information about violated
        (or potentially violated) constraints.
        """
        return run_diagnostics(self.objective, self.objective.new_weights,
                               self._cvx_objective, self._constraint_map)

    @property
    def status(self):
        """
        str

        String indicating the status of the optimization.
        
        """
        return self.prob.status

    @property
    def success(self):
        """
        class:bool

        True if the optimization successfully produced a result.
        """
        return self.prob.status == 'optimal'
