"""
优化结果模型

尚未完成有关限制条件冲突之类的检查
"""
import collections
import numpy as np
import pandas as pd


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
    def __init__(self, prob, objective, old_weights):
        self.prob = prob
        self.objective = objective
        self._old_weights = old_weights

    def raise_for_status(self):
        """
        Raise an error if the optimization did not succeed.
        """
        if self.prob.status != 'optimal':
            raise NotImplementedError('未成功优化')

    def _diagnostics_str(self):
        prob = self.prob
        data = collections.OrderedDict({
            '求解状态':
            '成功' if prob.status == 'optimal' else '失败',
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
        if not self.success:
            print('求解失败')
            return
        print(self._diagnostics_str())

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
        pandas.DataFrame 或 None
            
        New optimal weights, or None if the optimization failed.

        """
        if not self.success:
            return None
        long_w = self.objective.long_weights_value
        short_w = self.objective.short_weights_value
        return pd.DataFrame(
            {
                'long': long_w.values,
                'short': short_w.values,
                'total': (long_w - short_w).values
            },
            index=long_w.index)

    @property
    def net_weights(self):
        """
        净权重
        pandas.Series 或 None
            
        New optimal weights, or None if the optimization failed.

        """
        if not self.success:
            return None
        long_w = self.objective.long_weights_value
        short_w = self.objective.short_weights_value
        return long_w + short_w

    @property
    def abs_weights(self):
        """
        总权重
        pandas.Series 或 None
        
        
        New optimal weights, or None if the optimization failed.

        """
        return self.objective.weights_value

    @property
    def long_weights(self):
        """
        多头总权重
        pandas.Series 或 None
        
        
        New optimal weights, or None if the optimization failed.

        """
        return self.objective.long_weights_value

    @property
    def short_weights(self):
        """
        空头总权重
        pandas.Series 或 None
        
        
        New optimal weights, or None if the optimization failed.

        """
        return self.objective.short_weights_value

    @property
    def diagnostics(self):
        """
        zipline.optimize.Diagnostics

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