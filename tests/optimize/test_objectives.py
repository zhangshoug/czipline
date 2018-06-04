import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import cvxpy as cvx
from zipline.optimize import MaximizeAlpha, TargetWeights


class TestObjectives(unittest.TestCase):
    def setUp(self):
        stocks = ['000001', '000002', '000003', '000004', '000005']
        alphas = pd.Series([0.1, 0.2, 0.2, -0.1, -0.3], index=stocks)
        weights = pd.Series([-0.2, -0.2, 0.5, 0.5, 0.4], index=stocks)
        self.stocks = stocks
        self.alphas = alphas
        self.weights = weights

    def check(self, prob, obj, desired_value, desired_weights, init_w_s=None):
        prob.solve()
        # print('求解状态', prob.status)
        # print('最优解', prob.value)
        # print('权重\n', obj.new_weights_value)
        # print('初始权重\n', init_w_s)
        self.assertAlmostEqual(desired_value, prob.value, 2)
        assert_series_equal(
            obj.new_weights_value, desired_weights, check_less_precise=4)

    def test_maximizeAlpha_1(self):
        """限定多头权重和为1，不允许空头"""
        obj = MaximizeAlpha(self.alphas)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [
            cvx.sum(cvx.abs(w)) <= 1.0,
            w >= 0.0,
        ]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.2
        desired_weights = pd.Series([0, 0.5, 0.5, 0, 0], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_maximizeAlpha_2(self):
        """限定空头权重和为1，不允许多头"""
        obj = MaximizeAlpha(self.alphas)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [
            cvx.sum(cvx.abs(w)) <= 1.0,
            w <= 0.0,
        ]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.3
        desired_weights = pd.Series([0, 0., 0., 0, -1.0], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_maximizeAlpha_3(self):
        """限定单个权重值"""
        obj = MaximizeAlpha(self.alphas)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [
            cvx.abs(w) <= 0.2,
        ]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.18
        desired_weights = pd.Series(
            [0.2, 0.2, 0.2, -0.2, -0.2], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_maximizeAlpha_4(self):
        """限定总权重值，且单个空头权重不得小于-0.2，即最大空头绝对值不得超过0.2"""
        obj = MaximizeAlpha(self.alphas)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [
            cvx.sum(cvx.abs(w)) <= 1.0,
            w >= -0.2,
        ]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.22
        desired_weights = pd.Series(
            [0., 0.4, 0.4, -0., -0.2], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_maximizeAlpha_5(self):
        """
        单纯限定总权重，而不限定多头或空头值
        """
        obj = MaximizeAlpha(self.alphas)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [
            cvx.sum(cvx.abs(w)) <= 1.0,
            # w >= -0.2,
        ]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.30
        desired_weights = pd.Series([0., 0., 0., -0., -1.], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_maximizeAlpha_6(self):
        """
        如果限定总权重，且限定多头及多头权重值

        最会依次取正收益，直至权重和为限定值
        """
        obj = MaximizeAlpha(self.alphas)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [
            cvx.sum(cvx.abs(w)) <= 1.0,
            w >= 0,
        ]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.20
        desired_weights = pd.Series(
            [0., 0.5, 0.5, -0., -0.], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_TargetWeights_1(self):
        """
        不做任何限定，原样复制目标权重
        """
        obj = TargetWeights(self.weights)
        cvx_objective = obj.to_cvxpy(None)
        constraints = []
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.0
        desired_weights = pd.Series(
            [-0.2, -0.2, 0.5, 0.5, 0.4], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_TargetWeights_2(self):
        """
        限定为空头
        """
        obj = TargetWeights(self.weights)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [w <= 0]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.66
        desired_weights = pd.Series(
            [-0.2, -0.2, 0., 0., 0.], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)

    def test_TargetWeights_3(self):
        """
        限定为多头
        """
        obj = TargetWeights(self.weights)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        constraints = [w >= 0]
        prob = cvx.Problem(cvx_objective, constraints)
        desired_value = 0.08
        desired_weights = pd.Series([0., 0., 0.5, 0.5, 0.4], index=self.stocks)
        self.check(prob, obj, desired_value, desired_weights)


if __name__ == '__main__':
    unittest.main()
