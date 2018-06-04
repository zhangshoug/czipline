import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal

import cvxpy as cvx
from zipline.optimize import MaximizeAlpha, TargetWeights
from zipline.optimize.constraints import (
    Basket, CannotHold, DollarNeutral, FactorExposure, FixedWeight, Frozen,
    LongOnly, MaxGrossExposure, NetExposure, NetGroupExposure, NotConstrained,
    NotExceed, NotLessThan, Pair, PositionConcentration, ReduceOnly, ShortOnly)


class TestObjectives(unittest.TestCase):
    def setUp(self):
        stocks = [str(i).zfill(6) for i in range(1, 8)]
        alphas = pd.Series([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], index=stocks)
        # 用于调整特定目标
        # alphas = pd.Series([-0.2, -0.1, -0.1, 0, 0.1, 0.2, 0.3], index=stocks)
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
        factors = ['f1', 'f2', 'f3']
        data = [[3, 2, 3], [5, 1, 5], [2, 3, 1], [3, 5, 2], [4, 2, 3],
                [3, 4, 4]]  #, [2, 5, 3]],
        loadings = pd.DataFrame(data=data, index=stocks[:6], columns=factors)
        min_exposures = pd.Series([2, 1, 2], index=factors)
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

        # 假设初始权重股票数量少于目标权重的股票数量(4只股票)
        init_weights = pd.Series([-0.6, -0.1, 0.1, 0.55], index=stocks[:4])
        self.init_weights = init_weights

    def check(self, cons_obj, desired_value, desired_weights, init_w_s=None):
        # 基础限制：单个权重不超过0.5，空头不超过-1.0，总体不超过1.5
        obj = MaximizeAlpha(self.alphas)
        cvx_objective = obj.to_cvxpy(None)
        w = obj.new_weights
        w_s = obj.new_weights_series
        constraints = [
            con for con in cons_obj.gen_constraints(w, w_s, init_w_s)
        ]
        constraints += [cvx.norm(w, 1) <= 1.5, w <= 0.5, w >= -1.0]
        prob = cvx.Problem(cvx_objective, constraints)
        prob.solve()
        # print('求解状态', prob.status)
        # print('最优解', prob.value)
        # print('权重序列\n', obj.new_weights_value)
        # print('初始权重序列\n', init_w_s)
        self.assertAlmostEqual(desired_value, prob.value, 2)
        assert_series_equal(
            obj.new_weights_value,
            desired_weights,
            check_exact=False,
            check_less_precise=3)  # 精度：小数点后3位

    def test_MaxGrossExposure(self):
        cons_obj = MaxGrossExposure(1.5)
        desired_value = 0.45
        desired_weights = pd.Series([-1.0] + [0.0] * 5 + [0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_NetExposure(self):
        cons_obj = NetExposure(-0.1, 0.1)
        desired_value = 0.43
        desired_weights = pd.Series(
            [-0.80] + [0.0] * 4 + [0.2, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_DollarNeutral(self):
        cons_obj = DollarNeutral()
        desired_value = 0.33
        desired_weights = pd.Series(
            [-0.25] + [0.0] * 2 + [0.25, 0, 0.5, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, self.init_weights)

    def test_NetGroupExposure_1(self):
        """检验输入"""
        with self.assertRaises(TypeError):
            NetGroupExposure('a', 'b', 'c')
        # with self.assertRaises(AssertionError):
        #     labels = {'AAPL': 'TECH', 'MSFT': 'TECH', 'TSLA': 'CC', 'GM': 'CC'}
        #     # 限定不平衡
        #     min_exposures = {'TECH': -0.5, 'CC': -0.25}
        #     max_exposures = {'TECH': 0.5}
        #     NetGroupExposure(labels, min_exposures, max_exposures)
        with self.assertRaises(AssertionError):
            labels = {'AAPL': 'TECH', 'MSFT': 'TECH', 'TSLA': 'CC', 'GM': 'CC'}
            min_exposures = {'TECH': -0.5, 'CC': -0.25}
            max_exposures = {'TECH': 0.5, 'CC': -0.5}  # 最大值小于最小值
            ne = NetGroupExposure(labels, min_exposures, max_exposures)
            self.check(ne, 1, 1, 1)

    def test_NetGroupExposure_2(self):
        """分组净多头单边限制"""
        min_weights = pd.Series(
            [0.0, 0.1, 0.2, 0.3], index=self.min_weights.keys())
        cons_obj = NetGroupExposure(self.labels, min_weights, self.max_weights)
        desired_value = 0.21
        # 精度累死
        desired_weights = pd.Series(
            [-0.1612, 0.1612, -0.1207, 0.2207, -0.068, 0.368, 0.400],
            index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_NetGroupExposure_3(self):
        """分组净空头单边限制"""
        # 最大值负数
        max_weights = pd.Series([-0.05] * 4, index=self.max_weights.keys())
        cons_obj = NetGroupExposure(self.labels, self.min_weights, max_weights)
        desired_value = 0.08
        desired_weights = pd.Series(
            [-0.2908, 0.1908, -0.3606, 0.1606, -0.2485, 0.1985, -0.05],
            index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_NetGroupExposure_4(self):
        """分组双边限制(下限为负，上限为正)"""
        cons_obj = NetGroupExposure(self.labels, self.min_weights,
                                    self.max_weights)
        desired_value = 0.25
        desired_weights = pd.Series(
            [-0.2028, 0.1029, -0.284156, 0.084156, -0.062988, 0.3629, 0.4],
            index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_NetGroupExposure_5(self):
        """取消单组最大或最小限制"""
        min_weights = pd.Series(
            [-0.3, 0.1, 0., 0.3], index=self.min_weights.keys())
        max_weights = pd.Series([0.35] * 4, index=self.max_weights.keys())
        # 取消A组最小限制
        min_weights.loc['A'] = NotConstrained  # 可多分配空头
        # 取消C组最大限制
        max_weights.loc['C'] = NotConstrained

        cons_obj = NetGroupExposure(self.labels, min_weights, max_weights)
        desired_value = 0.41
        desired_weights = pd.Series(
            [-1, -0.000197, -0., 0.1, 0, 0.0498, 0.35], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_PositionConcentration_0(self):
        """验证输入及配置"""
        min_weights = pd.Series([-0.2, 0.1, 0.4], index=['000001', '000003','000004'])
        max_weights = pd.Series([-0.1, 0.4, 0.3], index=['000001', '000007','000004'])
        with self.assertRaises(ValueError):
            cons_obj = PositionConcentration(min_weights, max_weights)
            desired_value = 0.19
            desired_weights = pd.Series(
                [-0.2, 0, -0.1] + [0.0] * 3 + [0.4], index=self.stocks)
            self.check(cons_obj, desired_value, desired_weights)

    def test_PositionConcentration_1(self):
        """指定所有股票权重"""
        min_weights = pd.Series(
            [0.1, -0.1, 0.1, 0.2, -0.1, 0.2, 0.3], index=self.stocks)
        max_weights = pd.Series(
            [0.3, 0.4, 0.25, 0.25, 0.3, 0.25, 0.4], index=self.stocks)
        cons_obj = PositionConcentration(min_weights, max_weights)
        desired_value = 0.18
        desired_weights = pd.Series(
            [0.1, -0.1, 0.1, 0.2073, 0.3, 0.25, 0.4], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_PositionConcentration_2(self):
        """部分指定权重"""
        min_weights = pd.Series(
            [-0.4, -0.3, 0.1], index=['000001', '000003', '000004'])
        max_weights = pd.Series([0.1, 0.3], index=['000002', '000004'])
        cons_obj = PositionConcentration(min_weights, max_weights)
        desired_value = 0.37
        desired_weights = pd.Series(
            [-0.4, 0, 0, 0.1, 0.0] + [0.5] * 2, index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_PositionConcentration_3(self):
        """多头权重"""
        min_weights = pd.Series()
        max_weights = pd.Series()
        cons_obj = PositionConcentration(min_weights, max_weights, 0.1, 0.5)
        desired_value = 0.20
        desired_weights = pd.Series([0.1] * 5 + [0.5] * 2, index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, self.init_weights)

    def test_PositionConcentration_4(self):
        """空头权重"""
        min_weights = pd.Series()
        max_weights = pd.Series()
        cons_obj = PositionConcentration(min_weights, max_weights, -0.5, -0.1)
        desired_value = 0.37
        desired_weights = pd.Series(
            [-0.5, -0.2107, -0.1, -0.1, 0, 0.089275, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, self.init_weights)

    def test_PositionConcentration_5(self):
        """最小最大分别位于0值二端"""
        min_weights = pd.Series()
        max_weights = pd.Series()
        cons_obj = PositionConcentration(min_weights, max_weights, -0.5, 0.2)
        desired_value = 0.40
        desired_weights = pd.Series(
            [-0.5, -0.2853, 0, 0, 0, 0.2146, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, self.init_weights)

    def test_FactorExposure(self):
        cons_obj = FactorExposure(self.loadings, self.min_exposures,
                                  self.max_exposures)
        desired_value = 0.34
        desired_weights = pd.Series(
            [-0.2142] + [0.0] * 3 + [0.2857, 0.5, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_Pair(self):
        init_w_s = pd.Series(
            [-0.5, 0, 0, 0, 0, 0.5, 0], index=self.stocks) / 10.
        cons_obj = Pair('000006', '000001', 2)
        desired_value = 0.37
        desired_weights = pd.Series(
            [-0.25] * 2 + [0.0] * 3 + [0.5, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, init_w_s)

    def test_Basket_1(self):
        """容易出错的负数权重输入验证"""
        with self.assertRaises(AssertionError):
            Basket(['000002', '000003'], -0.1, -0.2)

    def test_Basket_2(self):
        cons_obj = Basket(['000002', '000003'], -0.2, -0.1)
        # 2/3为负收益，使用限定空头净敞口，检验是否挤占1的额度
        desired_value = 0.42
        # 其实达成目标收益率有无穷组合，只要第1和第7的权重和等于1.3
        # 不知道底层如何分配?
        # 初步判断：多头限定0.5最大，而空头1.0最大，按比例分配给空头近2倍于多头的量
        # 目前只是确定，多次允许结果相同，倒是不会随机组合
        desired_weights = pd.Series(
            [-0.863, -0.1, -0.1] + [0.0, 0, 0, 0.4367], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_Frozen_1(self):
        """无论空头与多头，000004都没有收益，但占用总体额度，测试是否保留权重"""
        cons_obj = Frozen(['000004'])
        init_w_s = pd.Series([0, 0, -3, 3, 2, 4, 0], index=self.stocks) / 10.
        desired_value = 0.36
        desired_weights = pd.Series(
            [-0.929683,0,0,0.3, 0.0, 0.0, 0.2703], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights,init_w_s)

    def test_Frozen_2(self):
        """固定多个"""
        cons_obj = Frozen(['000001', '000002'])
        init_w_s = pd.Series([2, -1, -3, 3, 2, 4, 0], index=self.stocks) / 10.
        desired_value = 0.23
        desired_weights = pd.Series(
            [0.2, -0.1, -0.1187, 0., 0.08126, 0.5, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, init_w_s)

    def test_ReduceOnly_1(self):
        """测试不得增加空头头寸"""
        cons_obj = ReduceOnly(['000001', '000002'])
        # 尽管可以增加空头权重提高收益率，但第0和1项受限于不得增加权重
        init_w_s = pd.Series([-5, 0, -3, 0, 2, 4, 5], index=self.stocks) / 10.
        desired_value = 0.40
        desired_weights = pd.Series(
            [-0.5] + [0.0] * 4 + [0.5, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, init_w_s)

    def test_ReduceOnly_1(self):
        """测试不得增加多头与空头寸"""
        cons_obj = ReduceOnly(['000001','000007'])
        init_w_s = pd.Series([-5, 0, -2, 0, 2, 4, 5], index=self.stocks) / 10.
        desired_value = 0.40
        desired_weights = pd.Series(
            [-0.5,-0.3405] + [0.0] * 3 + [0.1594, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights, init_w_s)

    def test_LongOnly(self):
        cons_obj = LongOnly(self.stocks)
        desired_value = 0.30
        desired_weights = pd.Series([0.0] * 4 + [0.5] * 3, index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_ShortOnly(self):
        cons_obj = ShortOnly(self.stocks[:3])
        desired_value = 0.45
        desired_weights = pd.Series(
            [-1.0] + [0] * 5 + [0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_FixedWeight(self):
        cons_obj = FixedWeight(self.stocks[-2], 0.4)
        desired_value = 0.41
        desired_weights = pd.Series(
            [-0.8581] + [0] * 4 + [0.4, 0.2418], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_CannotHold_1(self):
        cons_obj = CannotHold(self.stocks[2:4])
        desired_value = 0.45
        desired_weights = pd.Series(
            [-1.0] + [0] * 5 + [0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_CannotHold_2(self):
        """测试输入单个资产"""
        cons_obj = CannotHold([self.stocks[2]])
        desired_value = 0.45
        desired_weights = pd.Series(
            [-1.0] + [0] * 5 + [0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_NotExceed(self):
        cons_obj = NotExceed(0.4)
        desired_value = 0.38
        desired_weights = pd.Series(
            [-0.4, -0.3596, 0, 0, 0, 0.3404, 0.4], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)

    def test_NotLessThan(self):
        cons_obj = NotLessThan(0.05)
        desired_value = 0.25
        desired_weights = pd.Series(
            [0.05, 0.05, 0.05, 0.05, 0.3, 0.5, 0.5], index=self.stocks)
        self.check(cons_obj, desired_value, desired_weights)


if __name__ == '__main__':
    unittest.main()
