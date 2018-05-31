import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from pandas.testing import assert_series_equal, assert_frame_equal
import cvxpy as cvx
from zipline.optimize import MaximizeAlpha, TargetWeights
from zipline.optimize.constraints import (
    CannotHold, FixedWeight, ShortOnly, LongOnly, ReduceOnly, Frozen, Basket,
    Pair, FactorExposure, DollarNeutral, MaxGrossExposure, NetExposure,
    NetGroupExposure, NotConstrained, PositionConcentration)


class TestObjectives(unittest.TestCase):
    def setUp(self):
        stocks = [str(i).zfill(6) for i in range(1, 8)]
        alphas = pd.Series([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], index=stocks)
        weights = pd.Series(alphas.values, index=stocks)
        labels = {
            '000001': 'A',
            '000002': 'A',
            '000003': 'B',
            '000004': 'B',
            '000005': 'C',
            '000006': 'C',
            '000007': 'D'
        }
        min_weights = {'A': -0.1, 'B': -0.2, 'C': -0.3, 'D': -0.4}
        max_weights = {'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4}
        factors = ['股本', '成长', '净利润']
        loadings = pd.DataFrame(
            data=[[3, 2, 3], [5, 1, 5], [2, 3, 1], [3, 5, 2], [4, 2, 3],
                  [3, 4, 4], [2, 5, 3]],
            index=stocks,
            columns=factors)
        min_exposures = pd.Series([2, 4, 2], index=factors)
        max_exposures = pd.Series([4, 5, 4], index=factors)
        self.stocks = stocks
        self.alphas = alphas
        self.weights = weights
        self.labels = labels
        self.min_weights = min_weights
        self.max_weights = max_weights

        self.loadings = loadings
        self.min_exposures = min_exposures
        self.max_exposures = max_exposures

    def check(self,
              cons_obj,
              value,
              desired_long,
              desired_short,
              init_w_s=None):
        # 基础限制：单个多头权重不超过0.5，空头不超过-1.0，总体不超过1.5
        obj = MaximizeAlpha(self.alphas)
        l_w = obj.long_w
        s_w = obj.short_w
        w = obj.w
        l_w_s = obj.long_weights_series
        s_w_s = obj.short_weights_series
        constraints = [
            con for con in cons_obj.gen_constraints(l_w, s_w, l_w_s, s_w_s,
                                                    init_w_s)
        ]
        constraints += [cvx.norm(w, 1) <= 1.5, l_w <= 0.5, s_w >= -1.0]
        prob = cvx.Problem(obj.objective, constraints)
        prob.solve()
        # print('求解状态', prob.status)
        # print('最优解', prob.value)
        # print('多头权重\n', obj.long_weights_value)
        # print('空头权重\n', obj.short_weights_value)
        # print('绝对值权重\n', obj.weights_value)
        # print('初始权重\n', init_w_s)
        self.assertAlmostEqual(value, prob.value, 2)
        assert_series_equal(
            obj.long_weights_value, desired_long, check_less_precise=4)
        assert_series_equal(
            obj.short_weights_value, desired_short, check_less_precise=4)

    def test_CannotHold(self):
        cons_obj = CannotHold(self.stocks[2:4])
        value = 0.45
        desired_long = pd.Series([0] * 6 + [0.5], index=self.stocks)
        desired_short = pd.Series([-1.0] + [0] * 6, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_FixedWeight(self):
        cons_obj = FixedWeight(self.stocks[-2], 0.4)
        value = 0.41
        desired_long = pd.Series([0] * 5 + [0.4, 0.2834], index=self.stocks)
        desired_short = pd.Series([-0.8166] + [0] * 6, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_ShortOnly(self):
        cons_obj = ShortOnly(self.stocks)
        value = 0.40
        desired_long = pd.Series([0.0] * 7, index=self.stocks)
        desired_short = pd.Series([-1.0, -0.5] + [0] * 5, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_LongOnly(self):
        cons_obj = LongOnly(self.stocks)
        value = 0.30
        desired_long = pd.Series([0.0] * 4 + [0.5] * 3, index=self.stocks)
        desired_short = pd.Series([0.0] * 7, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_ReduceOnly(self):
        cons_obj = ReduceOnly(['000001', '000002'])
        # 尽管可以增加空头权重提高收益率，但第0和1项受限于不得增加权重
        init_w_s = pd.Series([-5, 0, -3, 0, 2, 4, 5], index=self.stocks) / 10.
        value = 0.40
        desired_long = pd.Series(
            [0.0] * 4 + [0.0028, 0.5, 0.5], index=self.stocks)
        desired_short = pd.Series(
            [-0.495, -0.0, -0.0022] + [0.0] * 4, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short, init_w_s)

    def test_Frozen(self):
        cons_obj = Frozen(['000004'])
        # 无论空头与多头，000004都没有收益，但占用总体额度，测试是否保留权重
        init_w_s = pd.Series([0, 0, -3, 3, 2, 4, 0], index=self.stocks) / 10.
        value = 0.36
        desired_long = pd.Series(
            [0.0] * 3 + [0.3, 0.0, 0.0, 0.29], index=self.stocks)
        desired_short = pd.Series([-0.91] + [0.0] * 6, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short, init_w_s)

    def test_Basket(self):
        cons_obj = Basket(['000002', '000003'], -0.2, -0.1)
        # 2/3为负收益，使用限定空头净敞口，检验是否挤占1的额度
        value = 0.42
        desired_long = pd.Series([0.0] * 6 + [0.3262], index=self.stocks)
        desired_short = pd.Series(
            [-0.9738, -0.1, -0.1] + [0.0] * 4, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_Pair(self):
        cons_obj = Pair('000007', '000001', 0.4)
        value = 0.38
        desired_long = pd.Series([0.0] * 5 + [0.3199, 0.4], index=self.stocks)
        desired_short = pd.Series(
            [-0.4, -0.3801] + [0.0] * 5, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_FactorExposure(self):
        cons_obj = FactorExposure(self.loadings, self.min_exposures,
                                  self.max_exposures)
        value = 0.37
        desired_long = pd.Series(
            [0.0] * 4 + [0.1429, 0.5, 0.5], index=self.stocks)
        desired_short = pd.Series([-0.3571] + [0.0] * 6, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_PositionConcentration_1(self):
        """指定所有股票权重"""
        min_weights = pd.Series(
            [0.1, -0.1, 0.1, 0.2, -0.1, 0.2, 0.3], index=self.stocks)
        max_weights = pd.Series(
            [0.3, 0.4, 0.2, 0.25, 0.3, 0.25, 0.4], index=self.stocks)
        cons_obj = PositionConcentration(min_weights, max_weights)
        value = 0.18
        desired_long = pd.Series(
            [0.1, 0, 0.1, 0.2055, 0.3, 0.25, 0.4], index=self.stocks)
        desired_short = pd.Series([0, -0.1] + [0.0] * 5, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_PositionConcentration_2(self):
        """部分指定权重"""
        min_weights = pd.Series([-0.2, -0.1], index=['000001', '000003'])
        max_weights = pd.Series([-0.1, 0.4], index=['000001', '000007'])
        cons_obj = PositionConcentration(min_weights, max_weights)
        value = 0.19
        desired_long = pd.Series([0] * 6 + [0.4], index=self.stocks)
        desired_short = pd.Series(
            [-0.2, 0, -0.1] + [0.0] * 4, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_PositionConcentration_3(self):
        """多头权重"""
        min_weights = pd.Series()
        max_weights = pd.Series()
        cons_obj = PositionConcentration(min_weights, max_weights, 0.1, 0.5)
        value = 0.20
        desired_long = pd.Series([0.1] * 5 + [0.5] * 2, index=self.stocks)
        desired_short = pd.Series([0.] * 7, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_PositionConcentration_4(self):
        """空头权重"""
        min_weights = pd.Series()
        max_weights = pd.Series()
        cons_obj = PositionConcentration(min_weights, max_weights, -0.5, -0.1)
        value = 0.20
        desired_long = pd.Series([0.] * 7, index=self.stocks)
        desired_short = pd.Series([-0.5] * 2 + [-0.1] * 5, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_PositionConcentration_5(self):
        """最小最大分别位于0值二端"""
        min_weights = pd.Series()
        max_weights = pd.Series()
        cons_obj = PositionConcentration(min_weights, max_weights, -0.5, 0.2)
        value = 0.36
        desired_long = pd.Series(
            [0.] * 4 + [0.0467, 0.2, 0.2], index=self.stocks)
        desired_short = pd.Series(
            [-0.5] * 2 + [-0.0533] + [-0.] * 4, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_NetGroupExposure_1(self):
        """检验输入"""
        with self.assertRaises(TypeError):
            NetGroupExposure('a', 'b', 'c')
        with self.assertRaises(AssertionError):
            labels = {'AAPL': 'TECH', 'MSFT': 'TECH', 'TSLA': 'CC', 'GM': 'CC'}
            # 限定不平衡
            min_exposures = {'TECH': -0.5, 'CC': -0.25}
            max_exposures = {'TECH': 0.5}
            NetGroupExposure(labels, min_exposures, max_exposures)
        with self.assertRaises(AssertionError):
            labels = {'AAPL': 'TECH', 'MSFT': 'TECH', 'TSLA': 'CC', 'GM': 'CC'}
            min_exposures = {'TECH': -0.5, 'CC': -0.25}
            max_exposures = {'TECH': 0.5, 'CC': -0.5}  # 最大值小于最小值
            ne = NetGroupExposure(labels, min_exposures, max_exposures)
            self.check(ne, 1, 1, 1)

    def test_NetGroupExposure_2(self):
        """分组多头限制"""
        min_weights = pd.Series([0.05] * 4, index=self.min_weights.keys())
        cons_obj = NetGroupExposure(self.labels, min_weights, self.max_weights)
        value = 0.18
        desired_long = pd.Series(
            [0.05, 0.05, 0.05, 0.0888, 0.3, 0.3, 0.4], index=self.stocks)
        desired_short = pd.Series([-0.] * 7, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_NetGroupExposure_3(self):
        """分组空头限制"""
        # 最大值都为负数，意味着不允许多头
        max_weights = pd.Series([-0.05] * 4, index=self.max_weights.keys())
        cons_obj = NetGroupExposure(self.labels, self.min_weights, max_weights)
        value = 0.04
        desired_long = pd.Series([0.] * 7, index=self.stocks)
        desired_short = pd.Series(
            [-0.1, -0.1, -0.2, -0.1194] + [-0.05] * 3, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_NetGroupExposure_4(self):
        """分组下限为负，上限为正"""
        cons_obj = NetGroupExposure(self.labels, self.min_weights,
                                    self.max_weights)
        value = 0.28
        desired_long = pd.Series(
            [0.] * 3 + [0.0124, 0.3, 0.3, 0.4], index=self.stocks)
        desired_short = pd.Series(
            [-0.1, -0.1, -0.2, -0.0121] + [-0.0] * 3, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_MaxGrossExposure(self):
        cons_obj = MaxGrossExposure(1.5)
        value = 0.45
        desired_long = pd.Series([0.0] * 6 + [0.5], index=self.stocks)
        desired_short = pd.Series([-1.0] + [0.0] * 6, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_NetExposure(self):
        cons_obj = NetExposure(-0.1, 0.1)
        value = 0.43
        desired_long = pd.Series([0.0] * 5 + [0.2, 0.5], index=self.stocks)
        desired_short = pd.Series([-0.80] + [0.0] * 6, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)

    def test_DollarNeutral(self):
        cons_obj = DollarNeutral()
        value = 0.43
        desired_long = pd.Series([0.0] * 5 + [0.25, 0.5], index=self.stocks)
        desired_short = pd.Series([-0.75] + [0.0] * 6, index=self.stocks)
        self.check(cons_obj, value, desired_long, desired_short)


if __name__ == '__main__':
    unittest.main()
