import os
import inspect
from typing import Any, List, Tuple

import numba as nb
import numpy as np
import pandas as pd
from numba.extending import as_numba_type
from joblib import delayed, Parallel

from util import read_candle_feather, transform_candle_np_struct, transform_np_struct

from .simulator import SingleFutureSimulator, SingleInverseFutureSimulator

SIMULATOR_MAP = {'futures': SingleFutureSimulator, 'inverse_futures': SingleInverseFutureSimulator}


def get_backtest_runner(stra, simu_init):
    # 推断参数 numba 类型
    candle_type = np.dtype([('open', np.float64), ('high', np.float64), ('low', np.float64), ('close', np.float64),
                            ('volume', np.float64)])
    candle_type = nb.from_dtype(candle_type)

    factor_type = nb.from_dtype(np.dtype([(f, np.float64) for f in stra.FCOLS]))

    strategy_type = as_numba_type(stra.Strategy)
    return_type = as_numba_type(Tuple[nb.int64[:], nb.float64[:]])

    # 定义回测函数，这里会直接 jit 编译
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

    # 返回 jit 编译好的回测函数
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

    # 返回初始化模拟器的 jit 函数
    return simulator_initializer


def calc_factors(candles, stra_module, factor_params, start_date, end_date):
    df_fac: pd.DataFrame = stra_module.factor(candles, factor_params)  # 计算因子
    df_fac = df_fac[(df_fac['candle_begin_time'] >= start_date) & (df_fac['candle_begin_time'] <= end_date)]
    factors = transform_np_struct(df_fac[stra_module.FCOLS])  # 因子 numpy structured array
    candles = transform_candle_np_struct(df_fac)  # K线 numpy structured array
    return df_fac, candles, factors


def inject_strategy_params(stra_module, strategy_params, inject):
    # 为 strategy_params 注入 init_capital 和 face_value（如有需要）
    stra_init_params = inspect.signature(stra_module.Strategy.__init__).parameters
    for pname, pval in inject:
        if pname in stra_init_params:
            strategy_params[pname] = pval


class Backtester:

    def __init__(self, candle_paths, contract_type, simulator_params, stra_module):
        self.stra_module = stra_module  # 要回测的策略
        self.candle_paths = candle_paths

        self.candles = {itl: read_candle_feather(path) for itl, path in self.candle_paths.items()}  # 读取K线数据

        # jit 编译回测函数
        simu_init = get_simulator_initializer(contract_type, simulator_params)
        self.backtest_runner = get_backtest_runner(stra_module, simu_init)

    def run_detailed(self, start_date, end_date, init_capital, face_value, factor_params, strategy_params):
        df_fac, candles, factors = calc_factors(self.candles, self.stra_module, factor_params, start_date, end_date)
        pos, equity = self._run_backtest(candles, factors, init_capital, face_value, strategy_params)
        df_fac['pos'] = pos
        df_fac['equity'] = equity
        return df_fac

    def run_multi(self, start_date, end_date, init_capital, face_value, fparams_list, sparams_list):
        data = []
        for factor_params in fparams_list:
            _, candles, factors = calc_factors(self.candles, self.stra_module, factor_params, start_date, end_date)
            for strategy_params in sparams_list:
                pos, equity = self._run_backtest(candles, factors, init_capital, face_value, strategy_params)
                r = {'equity': equity[-1] / equity[0]}
                r.update(factor_params)
                r.update(strategy_params)
                data.append(r)
        return pd.DataFrame.from_records(data)

    def _run_backtest(self, candles, factors, init_capital, face_value, strategy_params):
        inject_strategy_params(self.stra_module, strategy_params, [('face_value', face_value),
                                                                   ('init_capital', init_capital)])
        strategy = self.stra_module.Strategy(**strategy_params)
        pos, equity = self.backtest_runner(candles, factors, strategy)
        return pos, equity


def run_gridsearch(stra_module,
                   candle_paths,
                   contract_type,
                   simulator_params,
                   start_date,
                   end_date,
                   init_capital,
                   face_value,
                   n_proc=None):
    if n_proc is None:
        n_proc = max(os.cpu_count() - 1, 1)

    fparams_list = stra_module.get_default_factor_params_list()
    sparams_list = stra_module.get_default_strategy_params_list()

    fparams_seqs = []
    n = len(fparams_list)
    j = 0
    for i in range(n_proc):
        n_tasks = n // n_proc
        if i < n % n_proc:
            n_tasks += 1
        fparams_seqs.append(fparams_list[j:j + n_tasks])
        j += n_tasks

    def _search(fl):
        backtester = Backtester(candle_paths=candle_paths,
                                contract_type=contract_type,
                                simulator_params=simulator_params,
                                stra_module=stra_module)
        df = backtester.run_multi(start_date=start_date,
                                  end_date=end_date,
                                  init_capital=init_capital,
                                  face_value=face_value,
                                  fparams_list=fl,
                                  sparams_list=sparams_list)
        return df

    dfs = Parallel(n_jobs=n_proc)(delayed(_search)(fl) for fl in fparams_seqs)
    return pd.concat(dfs, ignore_index=True)