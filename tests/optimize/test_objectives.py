import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_array_equal

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

    def test_maximizeAlpha_1(self):
        """限定多头权重和为1，不允许空头"""
        obj = MaximizeAlpha(self.alphas)
        l_w = obj.long_w
        s_w = obj.short_w
        constraints = [
            cvx.sum(cvx.abs(l_w)) <= 1.0,
            cvx.sum(s_w) >= 0.0,
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()
        desired_l_w = np.array([0, 0.5, 0.5, 0, 0])
        assert_almost_equal(prob.value, self.alphas.max().sum())
        assert_almost_equal(desired_l_w, l_w.value)

    def test_maximizeAlpha_2(self):
        """限定空头权重和为1，不允许多头"""
        obj = MaximizeAlpha(self.alphas)
        l_w = obj.long_w
        s_w = obj.short_w
        constraints = [
            cvx.sum(cvx.abs(s_w)) <= 1.0,
            cvx.sum(l_w) == 0.0,
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()
        desired_s_w = np.array([0, 0., 0., 0, -1.0])
        assert_almost_equal(prob.value, -self.alphas.min())
        assert_almost_equal(desired_s_w, s_w.value)

    def test_maximizeAlpha_3(self):
        """限定单个权重值"""
        obj = MaximizeAlpha(self.alphas)
        l_w = obj.long_w
        s_w = obj.short_w
        constraints = [
            l_w <= 0.2,
            s_w >= -0.2,
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()
        assert_almost_equal(prob.value, self.alphas.abs().sum() * 0.2)
        desired_l_w = np.array([0.2, 0.2, 0.2, 0, 0])
        assert_almost_equal(desired_l_w, l_w.value)
        desired_s_w = np.array([0., 0., 0., -0.2, -0.2])
        assert_almost_equal(desired_s_w, s_w.value)

    def test_maximizeAlpha_4(self):
        """限定总权重值"""
        obj = MaximizeAlpha(self.alphas)
        l_w = obj.long_w
        s_w = obj.short_w
        w = obj.w
        constraints = [
            cvx.sum(w) <= 1.0,
            s_w >= -0.2,
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()
        assert_almost_equal(prob.value, 0.22)
        # 绝对值权重
        desired_w = np.array([0., 0.4, 0.4, 0.0, 0.2])
        assert_almost_equal(desired_w, w.value)

        desired_l_w = np.array([0., 0.4, 0.4, 0, 0])
        assert_almost_equal(desired_l_w, l_w.value)
        desired_s_w = np.array([0., 0., 0., 0., -0.2])
        assert_almost_equal(desired_s_w, s_w.value)

    def test_maximizeAlpha_5(self):
        """
        如果单纯限定总权重，而不限定多头或空头值

        则取max(正数最大值,负数绝对值最大值)

        最大者权重和为1(存在多个平均，只有一个权重为1)
        """
        obj = MaximizeAlpha(self.alphas)
        l_w = obj.long_w
        s_w = obj.short_w
        w = obj.w
        constraints = [
            cvx.sum(cvx.abs(w)) <= 1.0,
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()

        assert_almost_equal(prob.value, 0.30)
        # 绝对值权重
        desired_w = np.array([0., 0.0, 0.0, 0.0, 1.0])
        assert_almost_equal(desired_w, w.value)

        desired_l_w = np.array([0., 0., 0., 0, 0])
        assert_almost_equal(desired_l_w, l_w.value)
        desired_s_w = np.array([0., 0., 0., 0., -1.0])
        assert_almost_equal(desired_s_w, s_w.value)

    def test_maximizeAlpha_6(self):
        """
        如果限定总权重，且限定多头及多头权重值

        最会依次取正收益，直至权重和为限定值
        """
        obj = MaximizeAlpha(self.alphas)
        l_w = obj.long_w
        s_w = obj.short_w
        w = obj.w
        constraints = [
            cvx.sum(w) <= 1.0,
            s_w >= 0.0,  # 必须限定不允许空头，否则会把空头最大的因子权重赋值为1
            l_w <= 0.2
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()

        assert_almost_equal(prob.value, 0.1, 2)
        # 绝对值权重
        desired_w = np.array([0.2, 0.2, 0.2, 0.0, 0.0])
        assert_almost_equal(desired_w, w.value, 2)

        desired_l_w = np.array([0.2, 0.2, 0.2, 0, 0])
        assert_almost_equal(desired_l_w, l_w.value, 2)
        desired_s_w = np.array([0., 0., 0., 0., 0])
        assert_almost_equal(desired_s_w, s_w.value, 2)

    def test_TargetWeights_1(self):
        """
        不做任何限定，原样复制目标权重
        """
        obj = TargetWeights(self.weights)
        l_w = obj.long_w
        s_w = obj.short_w
        w = obj.w
        constraints = []
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()

        assert_almost_equal(prob.value, 0.)
        t_w = self.weights.values
        desired_l_w, desired_s_w = np.array(t_w), np.array(t_w)
        # 绝对值权重
        assert_almost_equal(np.abs(t_w), w.value, 3)
        # 多头权重
        desired_l_w[desired_l_w < 0] = 0
        assert_almost_equal(desired_l_w, l_w.value, 3)
        # 空头权重
        desired_s_w[desired_s_w > 0] = 0
        assert_almost_equal(desired_s_w, s_w.value, 3)

    def test_TargetWeights_2(self):
        """
        限定为多头
        """
        obj = TargetWeights(self.weights)
        l_w = obj.long_w
        s_w = obj.short_w
        w = obj.w
        constraints = [
            # l_w <= 1.0,
            s_w >= 0.0,
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()

        t_w = self.weights.values
        desired_l_w, desired_s_w = np.array(t_w), np.array(t_w)
        desired_l_w[desired_l_w < 0] = 0
        desired_s_w = 0

        # 绝对值权重
        desired_w = np.abs(desired_l_w)
        assert_almost_equal(desired_w, w.value)

        assert_almost_equal(desired_l_w, l_w.value)

        assert_almost_equal(desired_s_w, s_w.value)

    def test_TargetWeights_3(self):
        """
        限定为空头
        """
        obj = TargetWeights(self.weights)
        l_w = obj.long_w
        s_w = obj.short_w
        w = obj.w
        constraints = [
            l_w <= 0.0,
            # s_w <= 0.0,
        ]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()

        t_w = self.weights.values
        desired_l_w, desired_s_w = np.array(t_w), np.array(t_w)
        desired_l_w = 0
        desired_s_w[desired_s_w > 0] = 0

        # 绝对值权重
        desired_w = np.abs(desired_s_w)
        assert_almost_equal(desired_w, w.value)

        assert_almost_equal(desired_l_w, l_w.value)

        assert_almost_equal(desired_s_w, s_w.value)


if __name__ == '__main__':
    unittest.main()
