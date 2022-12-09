import numba as nb
import numpy as np
from numba.experimental import jitclass


@jitclass([
    ['equity', nb.float64],  # 账户权益，可以设置为 10万
    ['face_value', nb.float64],  # 合约面值，或最小下单单位
    ['comm_rate', nb.float64],  # 手续费/交易成本，可设置为万分之6
    ['liqui_rate', nb.float64],  # 爆仓保证金率，千分之5
    ['pre_close', nb.float64],  # 上根k线的收盘价
    ['pos', nb.int64],  # 当前持仓
    ['target_pos', nb.int64]  # 目标持仓
])
class SingleFutureSimulator:
    """
    正向合约回测模拟
    """

    def __init__(self, init_capital, face_value, comm_rate, liqui_rate, init_pos):
        self.equity = init_capital
        self.face_value = face_value
        self.comm_rate = comm_rate
        self.liqui_rate = liqui_rate

        self.pre_close = np.nan
        self.pos = int(init_pos)
        self.target_pos = int(init_pos)

    def adjust_pos(self, target_pos):
        self.target_pos = target_pos

    def simulate_bar(self, candle):
        op = candle['open']
        hi = candle['high']
        lo = candle['low']
        cl = candle['close']
        
        if np.isnan(self.pre_close):
            self.pre_close = op

        # K 线开盘时刻
        # 根据开盘价和前收盘价，结算当前账户权益
        self.equity += (op - self.pre_close) * self.face_value * self.pos

        # 当前需要调仓
        if self.target_pos != self.pos:
            delta = self.target_pos - self.pos  # 需要买入或卖出的合约数量
            self.equity -= abs(delta) * self.face_value * op * self.comm_rate  # 扣除手续费
            self.pos = self.target_pos  # 更新持仓

        # K 线当中
        price_min = lo if self.pos > 0 else hi  # 根据持仓方向，找出账户权益最低的价格
        equity_min = self.equity + (price_min - op) * self.pos * self.face_value  #  最低账户权益
        if self.pos == 0:
            margin_ratio_min = 1e8  # 空仓，设置保证金率为极大值
        else:
            margin_ratio_min = equity_min / (self.face_value * abs(self.pos) * price_min)  # 有持仓，计算最低保证金率
        if margin_ratio_min < self.liqui_rate + self.comm_rate:  # 爆仓
            self.equity = 1e-8  # 设置这一刻起的资金为极小值，防止除零错误
            self.pos = 0

        # K线收盘时刻
        self.equity += (cl - op) * self.face_value * self.pos  # 根据收盘价，结算账户权益
        self.pre_close = cl


@jitclass([
    ['equity', nb.float64],  # 账户权益，可以设置为 10 个
    ['face_value', nb.float64],  # 合约面值，通常为 10 或 100 美元
    ['comm_rate', nb.float64],  # 手续费/交易成本，可设置为万分之6
    ['liqui_rate', nb.float64],  # 爆仓保证金率，千分之5
    ['pre_close', nb.float64],  # 上根k线的收盘价
    ['pos', nb.int64],  # 当前持仓
    ['target_pos', nb.int64]  # 目标持仓
])
class SingleInverseFutureSimulator:
    """
    反向合约回测模拟
    """

    def __init__(self, init_capital, face_value, comm_rate, liqui_rate, init_pos):
        self.equity = init_capital
        self.face_value = face_value
        self.comm_rate = comm_rate
        self.liqui_rate = liqui_rate

        self.pre_close = np.nan
        self.pos = int(init_pos)
        self.target_pos = int(init_pos)

    def adjust_pos(self, target_pos):
        self.target_pos = target_pos

    def simulate_bar(self, candle):
        op = candle['open']
        hi = candle['high']
        lo = candle['low']
        cl = candle['close']
        if np.isnan(self.pre_close):
            self.pre_close = op

        # K 线开盘时刻
        # 根据开盘价和前收盘价，结算当前账户权益
        self.equity += (1 / self.pre_close - 1 / op) * self.face_value * self.pos

        if self.target_pos != self.pos:  # 当前需要调仓
            delta = self.target_pos - self.pos  # 需要买入或卖出的合约数量
            self.equity -= abs(delta) * self.face_value / op * self.comm_rate  # 扣除手续费
            self.pos = self.target_pos  # 更新持仓

        # K 线当中
        price_min = lo if self.pos > 0 else hi  # 根据持仓方向，找出账户权益最低的价格
        equity_min = self.equity + (1 / op - 1 / price_min) * self.pos * self.face_value  # 最低账户权益
        if self.pos == 0:
            margin_ratio_min = 1e8  # 空仓，设置保证金率为极大值
        else:
            margin_ratio_min = price_min * equity_min / (self.face_value * abs(self.pos))  # 有持仓，计算最低保证金率
        if margin_ratio_min < self.liqui_rate + self.comm_rate:  # 爆仓
            self.equity = 1e-8  # 设置这一刻起的资金为极小值，防止除零错误
            self.pos = 0

        # K线收盘时刻
        self.equity += (1 / op - 1 / cl) * self.pos * self.face_value  # 根据收盘价，结算账户权益
        self.pre_close = cl