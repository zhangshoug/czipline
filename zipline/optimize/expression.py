"""
投资组合头寸与权重
+ 投资组合包含n个股票及现金账户
+ t = 1,..., T
+ h_t代表时间t的持仓头寸(以金额为单位而不是数量)
+ v_t代表组合净值
+ w_t = h_t / v_t

交易与交易后投资组合
+ u_t代表交易金额(含现金)
+ 假设在时间点t交易，则交易后投资组合为h_t + u_t
+ z_t = u_t / v_t
"""

from abc import ABCMeta, abstractmethod


class Expression(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def weight_expr(self, t, w_plus, z, value):
        """返回时点t的成本估计"""
        pass

    def weight_expr_ahead(self, t, tau, w_plus, z, value):
        """返回tau时点成本在t时点的估计"""
        return self.weight_expr(t, w_plus, z, value)

    def value_expr(self, t, h_plus, u):
        """
        返回时点t的表达式，用值表示
        如果在模拟器中使用该项，则应重写
        """
        return sum(h_plus) * self.weight_expr(t, h_plus / sum(h_plus),
                                              u / sum(u), sum(h_plus))
