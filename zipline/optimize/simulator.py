"""
模拟器

简化
1. 只需要运行一期
2. 假设交易成本、持有成本为0
"""

import copy
import logbook
import time

import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx

from .returns import MultipleReturnsForecasts

from .result import SimulationResult
from .costs import BaseCost

logger = logbook.Logger('投资组合优化')


class MarketSimulator():
    def __init__(self,
                 market_returns,
                 costs=[],
                 market_volumes=None,
                 cash_key='cash'):
        """Provide market returns object and cost objects."""
        # market_returns, assets = asset_to_symbol(market_returns, cash_key)
        self.market_returns = market_returns
        # self.assets = assets
        if market_volumes is not None:
            # market_volumes, _ = asset_to_symbol(market_volumes, cash_key)
            self.market_volumes = market_volumes[
                market_volumes.columns.difference(cash_key)]
        else:
            self.market_volumes = None

        self.costs = costs
        for cost in self.costs:
            assert (isinstance(cost, BaseCost))

        self.cash_key = cash_key

    def propagate(self, h, u, t):
        """Propagates the portfolio forward over time period t, given trades u.

        Args:
            h: 当前投资组合的序列 pandas Series object describing current portfolio
            u: 股票交易n个矢量(不含现金账户) n vector with the stock trades (not cash)
            t: current time

        Returns:
            h_next: 完成交易后的投资组合 portfolio after returns propagation
            u: 现金平衡后的交易矢量 trades vector with simulated cash balance
        """
        assert (u.index.equals(h.index))

        if self.market_volumes is not None:
            # don't trade if volume is null
            null_trades = self.market_volumes.columns[self.market_volumes.loc[
                t] == 0]
            if len(null_trades):
                logger.info('No trade condition for stocks %s on %s' %
                             (null_trades, t))
                u.loc[null_trades] = 0.

        hplus = h + u
        costs = [cost.value_expr(t, h_plus=hplus, u=u) for cost in self.costs]
        for cost in costs:
            assert (not pd.isnull(cost))
            assert (not np.isinf(cost))

        u[self.cash_key] = -sum(u[u.index != self.cash_key]) - sum(costs)
        hplus[self.cash_key] = h[self.cash_key] + u[self.cash_key]

        assert (hplus.index.sort_values().equals(
            self.market_returns.columns.sort_values()))
        h_next = self.market_returns.loc[t] * hplus + hplus

        assert (not h_next.isnull().values.any())
        assert (not u.isnull().values.any())
        return h_next, u

    def run_backtest(self, initial_portfolio, start_time, end_time, policy):
        """Backtest a single policy.
        """
        # initial_portfolio, _ = asset_to_symbol(initial_portfolio, self.cash_key)
        results = SimulationResult(
            initial_portfolio=copy.copy(initial_portfolio),
            policy=policy,
            cash_key=self.cash_key,
            simulator=self)
        h = initial_portfolio

        simulation_times = self.market_returns.index[
            (self.market_returns.index >= start_time)
            & (self.market_returns.index <= end_time)]
        logger.info('Backtest started, from %s to %s' %
                     (simulation_times[0], simulation_times[-1]))

        for t in simulation_times:
            logger.info('Getting trades at time %s' % t)
            start = time.time()
            try:
                u = policy.get_trades(h, t)
            except cvx.SolverError:
                logger.warning(
                    'Solver failed on timestamp %s. Default to no trades.' % t)
                u = pd.Series(index=h.index, data=0.)
            end = time.time()
            assert (not pd.isnull(u).any())
            results.log_policy(t, end - start)

            logger.info('Propagating portfolio at time %s' % t)
            start = time.time()
            h, u = self.propagate(h, u, t)
            end = time.time()
            assert (not h.isnull().values.any())
            results.log_simulation(
                t=t,
                u=u,
                h_next=h,
                risk_free_return=self.market_returns.loc[t, self.cash_key],
                exec_time=end - start)

        logger.info('Backtest ended, from %s to %s' % (simulation_times[0],
                                                        simulation_times[-1]))
        return results

    def run_multiple_backtest(self,
                              initial_portf,
                              start_time,
                              end_time,
                              policies,
                              parallel=True):
        """Backtest multiple policies.
        """

        def _run_backtest(policy):
            return self.run_backtest(
                initial_portf, start_time, end_time, policy, loglevel=loglevel)

        num_workers = min(multiprocess.cpu_count(), len(policies))
        if parallel:
            workers = multiprocess.Pool(num_workers)
            results = workers.map(_run_backtest, policies)
            workers.close()
            return results
        else:
            return list(map(_run_backtest, policies))

    def what_if(self, time, results, alt_policies, parallel=True):
        """Run alternative policies starting from given time.
        """
        # TODO fix
        initial_portf = copy.copy(results.h.loc[time])
        all_times = results.h.index
        alt_results = self.run_multiple_backtest(
            initial_portf, time, all_times[-1], alt_policies, parallel)
        for idx, alt_result in enumerate(alt_results):
            alt_result.h.loc[time] = results.h.loc[time]
            alt_result.h.sort_index(axis=0, inplace=True)
        return alt_results

    @staticmethod
    def reduce_signal_perturb(initial_weights, delta):
        """Compute matrix of perturbed weights given initial weights."""
        perturb_weights_matrix = \
            np.zeros((len(initial_weights), len(initial_weights)))
        for i in range(len(initial_weights)):
            perturb_weights_matrix[i, :] = initial_weights / \
                (1 - delta * initial_weights[i])
            perturb_weights_matrix[i, i] = (1 - delta) * initial_weights[i]
        return perturb_weights_matrix

    def attribute(self,
                  true_results,
                  policy,
                  selector=None,
                  delta=1,
                  fit="linear",
                  parallel=True):
        """Attributes returns over a period to individual alpha sources.

        Args:
            true_results: observed results.
            policy: the policy that achieved the returns.
                    Alpha model must be a stream.
            selector: A map from SimulationResult to time series.
            delta: the fractional deviation.
            fit: the type of fit to perform.
        Returns:
            A dict of alpha source to return series.
        """
        # Default selector looks at profits.
        if selector is None:

            def selector(result):
                return result.v - sum(result.initial_portfolio)

        alpha_stream = policy.return_forecast
        assert isinstance(alpha_stream, MultipleReturnsForecasts)
        times = true_results.h.index
        weights = alpha_stream.weights
        assert np.sum(weights) == 1
        alpha_sources = alpha_stream.alpha_sources
        num_sources = len(alpha_sources)
        Wmat = self.reduce_signal_perturb(weights, delta)
        perturb_pols = []
        for idx in range(len(alpha_sources)):
            new_pol = copy.copy(policy)
            new_pol.return_forecast = MultipleReturnsForecasts(
                alpha_sources, Wmat[idx, :])
            perturb_pols.append(new_pol)
        # Simulate
        p0 = true_results.initial_portfolio
        alt_results = self.run_multiple_backtest(p0, times[0], times[-1],
                                                 perturb_pols, parallel)
        # Attribute.
        true_arr = selector(true_results).values
        attr_times = selector(true_results).index
        Rmat = np.zeros((num_sources, len(attr_times)))
        for idx, result in enumerate(alt_results):
            Rmat[idx, :] = selector(result).values
        Pmat = cvx.Variable(num_sources, len(attr_times))
        if fit == "linear":
            prob = cvx.Problem(cvx.Minimize(0), [Wmat * Pmat == Rmat])
            prob.solve()
        elif fit == "least-squares":
            error = cvx.sum_squares(Wmat * Pmat - Rmat)
            prob = cvx.Problem(
                cvx.Minimize(error), [Pmat.T * weights == true_arr])
            prob.solve()
        else:
            raise Exception("Unknown fitting method.")
        # Dict of results.
        wmask = np.tile(weights[:, np.newaxis], (1, len(attr_times))).T
        data = pd.DataFrame(
            columns=[s.name for s in alpha_sources],
            index=attr_times,
            data=Pmat.value.T.A * wmask)
        data['residual'] = true_arr - np.matrix((weights * Pmat).value).A1
        data['RMS error'] = np.matrix(
            cvx.norm(Wmat * Pmat - Rmat, 2, axis=0).value).A1
        data['RMS error'] /= np.sqrt(num_sources)
        return data
