import numba as nb
import numpy as np
from .simulator import SingleFutureSimulator
from typing import List, Any
import pandas as pd


@nb.njit
def _run_backtest(candles, factor_mat, simu, stra):
    n = candles.shape[0]  # k 线数量
    pos_open = np.empty(n, dtype=np.int64)  # k 线 open 时刻的目标仓位，当根 K 线始终持有该仓位
    equity = np.empty(n, dtype=np.float64)  # k 线 close 时刻的账户权益

    # 遍历每根 k 线循环
    # print(n)
    for i in range(n):
        simu.simulate_bar(candles[i])  # 模拟调仓和 k 线内权益结算
        equity[i] = simu.equity  # 记录权益
        pos_open[i] = simu.pos  # 记录仓位
        target_pos = stra.on_bar(candles[i], factor_mat[i], simu.pos, simu.equity)  # 策略生成目标持仓
        simu.adjust_pos(target_pos)  # 记录目标持仓，下根 k 线 open 时刻调仓
    return pos_open, equity


def backtest_online(
        df: pd.DataFrame,  # 包含 k 线和因子的 dataframe
        simulator: SingleFutureSimulator,  # 模拟器，目前只支持 SingleFutureSimulator
        strategy: Any,  # 策略
        factor_columns: List[str]  # 因子的名称
):
    # 将 OHLCV K线转化为 Numpy 矩阵
    candles = df[['open', 'high', 'low', 'close', 'volume']].to_numpy()

    # 将因子转化为 Numpy 矩阵
    factor_mat = df[factor_columns].to_numpy()

    # 运行 jit 回测函数，获得仓位和权益
    pos, equity = _run_backtest(candles, factor_mat, simulator, strategy)
    df['pos'] = pos
    df['equity'] = equity