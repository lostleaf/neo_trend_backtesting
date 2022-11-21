import numba as nb
import numpy as np
from numba.experimental import jitclass


@jitclass([
    ['leverage', nb.float64],  # 杠杆率 
    ['face_value', nb.float64],  # 合约面值
    ['prev_upper', nb.float64],  # 上根k线上轨
    ['prev_lower', nb.float64],  # 上根k线下轨
    ['prev_median', nb.float64],  # 上根k线均线
    ['prev_close', nb.float64]  # 上根k线收盘价
])
class BollingStrategy:

    def __init__(self, leverage, face_value):
        self.leverage = leverage
        self.face_value = face_value

        self.prev_upper = np.nan
        self.prev_lower = np.nan
        self.prev_median = np.nan
        self.prev_close = np.nan

    def on_bar(self, candle, factors, pos, equity):
        op, hi, lo, cl, vol = candle
        upper, lower, median = factors

        # 默认保持原有仓位
        target_pos = pos

        if not np.isnan(self.prev_close):
            # 做空或无仓位，上穿上轨，做多
            if pos <= 0 and cl > upper and self.prev_close <= self.prev_upper:
                target_pos = int(equity * self.leverage / cl / self.face_value)

            # 做多或无仓位，下穿下轨，做空
            elif pos >= 0 and cl < lower and self.prev_close >= self.prev_lower:
                target_pos = -int(equity * self.leverage / cl / self.face_value)

            # 做多，下穿中轨，平仓
            elif pos > 0 and cl < median and self.prev_close >= self.prev_median:
                target_pos = 0

            # 做空，上穿中轨，平仓
            elif pos < 0 and cl > median and self.prev_close <= self.prev_median:
                target_pos = 0

        # 更新上根K线数据
        self.prev_upper = upper
        self.prev_lower = lower
        self.prev_close = cl
        self.prev_median = median

        return target_pos
