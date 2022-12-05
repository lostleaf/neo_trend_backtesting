import inspect
from typing import Any, List, Tuple

import numba as nb
import numpy as np
import pandas as pd
from numba.extending import as_numba_type

from util import read_candle_feather, transform_candle_np_struct, transform_np_struct

from .simulator import SingleFutureSimulator, SingleInverseFutureSimulator

SIMULATOR_MAP = {'futures': SingleFutureSimulator, 'inverse_futures': SingleInverseFutureSimulator}


def get_backtest_runner(stra, simu_init):
    candle_type = np.dtype([('open', np.float64), ('high', np.float64), ('low', np.float64), ('close', np.float64),
                            ('volume', np.float64)])
    candle_type = nb.from_dtype(candle_type)

    factor_type = nb.from_dtype(np.dtype([(f, np.float64) for f in stra.FCOLS]))

    strategy_type = as_numba_type(stra.Strategy)
    return_type = as_numba_type(Tuple[nb.int64[:], nb.float64[:]])

    @nb.njit(return_type(candle_type[:], factor_type[:], strategy_type))
    def _run_backtest(candles, factors, stra):
        n = candles.shape[0]  # k 线数量
        pos = np.empty(n, dtype=np.int64)  # K 线仓位
        equity = np.empty(n, dtype=np.float64)  # k 线 close 时刻的账户权益
        simu = simu_init()

        # 遍历每根 k 线循环
        # print(n)
        for i in range(n):
            simu.simulate_bar(candles[i])  # 开盘模拟调仓和 k 线内权益结算
            equity[i] = simu.equity  # 记录权益
            pos[i] = simu.pos  # 记录仓位
            target_pos = stra.on_bar(candles[i], factors[i], simu.pos, simu.equity)  # 策略生成目标持仓
            simu.adjust_pos(target_pos)  # 记录目标持仓，下根 k 线 open 时刻调仓
        return pos, equity

    return _run_backtest


def get_simulator_initializer(contract_type, simulator_params):
    simu_type = SIMULATOR_MAP[contract_type]
    init_capital = simulator_params['init_capital']
    face_value = simulator_params['face_value']
    comm_rate = simulator_params['comm_rate']
    liqui_rate = simulator_params['liqui_rate']
    init_pos = simulator_params['init_pos'] if 'init_pos' in simulator_params else 0

    @nb.njit
    def simulator_initializer():
        simulator = simu_type(init_capital, face_value, comm_rate, liqui_rate, init_pos)
        return simulator

    return simulator_initializer


def calc_factors(candles, stra_module, factor_params, start_date, end_date):
    df_fac: pd.DataFrame = stra_module.factor(candles, factor_params)
    df_fac = df_fac[(df_fac['candle_begin_time'] >= start_date) & (df_fac['candle_begin_time'] <= end_date)]
    factors = transform_np_struct(df_fac[stra_module.FCOLS])
    candles = transform_candle_np_struct(df_fac)
    return df_fac, candles, factors


def inject_strategy_params(stra_module, strategy_params, inject):
    stra_init_params = inspect.signature(stra_module.Strategy.__init__).parameters
    for pname, pval in inject:
        if pname in stra_init_params:
            strategy_params[pname] = pval


class Backtester:

    def __init__(self, candle_paths, contract_type, simulator_params, stra_module):
        self.stra_module = stra_module
        self.candles = {itl: read_candle_feather(path) for itl, path in candle_paths.items()}
        simu_init = get_simulator_initializer(contract_type, simulator_params)
        self.backtest_runner = get_backtest_runner(stra_module, simu_init)

    def run_detailed(self, start_date, end_date, init_capital, face_value, factor_params, strategy_params):
        df_fac, candles, factors = calc_factors(self.candles, self.stra_module, factor_params, start_date, end_date)
        pos, equity = self._run_backtest(candles, factors, init_capital, face_value, strategy_params)
        df_fac['pos'] = pos
        df_fac['equity'] = equity
        return df_fac

    def run_multi_sparams(self, start_date, end_date, init_capital, face_value, factor_params, sparams_list):
        _, candles, factors = calc_factors(self.candles, self.stra_module, factor_params, start_date, end_date)
        data = []
        for strategy_params in sparams_list:
            pos, equity = self._run_backtest(candles, factors, init_capital, face_value, strategy_params)
            r = {'equity': equity[-1]}
            r.update(factor_params)
            r.update(strategy_params)
            data.append(r)
        return pd.DataFrame.from_records(data)

    def run_gridsearch(self, start_date, end_date, init_capital, face_value, fparams_list, sparams_list):
        dfs = []
        for factor_params in fparams_list:
            dfs.append(
                self.run_multi_sparams(start_date, end_date, init_capital, face_value, factor_params, sparams_list))
        return pd.concat(dfs)

    def _run_backtest(self, candles, factors, init_capital, face_value, strategy_params):
        inject_strategy_params(self.stra_module, strategy_params, [('face_value', face_value),
                                                                   ('init_capital', init_capital)])
        strategy = self.stra_module.Strategy(**strategy_params)
        pos, equity = self.backtest_runner(candles, factors, strategy)
        return pos, equity
