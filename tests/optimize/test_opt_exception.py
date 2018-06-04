import pandas as pd
from zipline.optimize.result import (UnboundedObjective, InfeasibleConstraints,
                                     OptimizationFailed)
from zipline.optimize.constraints import (MaxGrossExposure, NetExposure,
                                          NotExceed, NotLessThan,
                                          DollarNeutral, FixedWeight)
from zipline.optimize.objectives import MaximizeAlpha
from zipline.optimize.core import run_optimization, calculate_optimal_portfolio
import unittest


class TestOptException(unittest.TestCase):
    def setUp(self):
        self.alphas = pd.Series(
            [0.1, 0.2, 0.3, -0.1], index=['a', 'b', 'c', 'd'])

    def test_input_check(self):
        with self.assertRaises(TypeError):
            MaximizeAlpha([0.1, 0.2, 0.3, -0.1])

    def test_unboundedObjective(self):
        objective = MaximizeAlpha(self.alphas)
        with self.assertRaises(UnboundedObjective):
            calculate_optimal_portfolio(objective, [])

    def test_infeasibleConstraints(self):
        objective = MaximizeAlpha(self.alphas)
        me = MaxGrossExposure(1.5)
        nl = NotLessThan(0.5)  # 单个权重不少于0.5与每个权重不得超过0.4矛盾
        ne = NotExceed(0.4)
        constraints = [me, ne, nl]
        with self.assertRaises(InfeasibleConstraints):
            calculate_optimal_portfolio(objective, constraints)

    def test_optimizationFailed(self):
        pass


if __name__ == '__main__':
    unittest.main()