import pandas as pd


def check_series_or_dict(x, name):
    """检查输入类型。输入若不是序列或者字典类型其中之一，则触发类型异常"""
    is_s = isinstance(x, pd.Series)
    is_d = isinstance(x, dict)
    if not (is_s or is_d):
        raise TypeError('参数名称{}类型错误。应为pd.Series或者dict'.format(name))


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


class OptimizationResult():
    def __init__(self, prob, new_weights, old_weights):
        self.prob = prob
        self._old_weights = old_weights
        self._new_weights = new_weights

    def raise_for_status(self):
        """
        Raise an error if the optimization did not succeed.
        """
        if self.prob.status != 'optimal':
            raise NotImplementedError('未成功优化')

    def print_diagnostics(self):
        """
        Print diagnostic information gathered during the optimization.
        """
        if not self.success:
            print('求解失败')
            return
        prob = self.prob
        objective = type(prob.objective)
        status = '成功' if prob.status == 'optimal' else '失败'
        num_iters = prob.solver_stats.num_iters
        setup_time = round(prob.solver_stats.setup_time, 6)
        solve_time = round(prob.solver_stats.solve_time, 6)
        solver_name = prob.solver_stats.solver_name
        msg_fmt = """
    目标：{}
    状态：{}
    最优解：{}
    迭代次数:{}
    启动时间：{}秒\t求解时间：{}秒
    求解器:{}
        """
        print(
            msg_fmt.format(objective, status, round(prob.value, 6), num_iters,
                           setup_time, solve_time, solver_name))

    @property
    def old_weights(self):
        """
        pandas.Series
        Portfolio weights before the optimization.
        """
        return self._old_weights

    @property
    def new_weights(self):
        """
        pandas.Series or None

        New optimal weights, or None if the optimization failed.

        """
        # 输出变量的值
        return self._new_weights.value

    @property
    def diagnostics(self):
        """
        quantopian.optimize.Diagnostics

        Object containing diagnostic information about violated
        (or potentially violated) constraints.
        """
        pass

    def status(self):
        """
        str

        String indicating the status of the optimization.
        
        """
        return self.prob.status

    def success(self):
        """
        class:bool

        True if the optimization successfully produced a result.
        """
        return self.prob.status == 'optimal'