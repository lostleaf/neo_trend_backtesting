import numba as nb
import numpy as np
import pandas as pd
import talib as ta
from numba.experimental import jitclass

FCOLS = ['upper', 'median', 'lower']


def factor(candles, params):
    # 取 K 线和参数
    df_1m = candles['1m']
    df_long = candles[params['itl']]
    n = params['n']
    b = params['b']

    # 计算布林带
    upper, median, lower = ta.BBANDS(df_long['close'], timeperiod=n, nbdevup=b, nbdevdn=b, matype=ta.MA_Type.SMA)
    df_fac = pd.DataFrame({'upper': upper, 'median': median, 'lower': lower}, index=df_long['candle_end_time'])

    # 将布林带填充到 1 分钟线
    df_fac = df_1m.join(df_fac, on='candle_end_time')
    for col in FCOLS:
        df_fac[col].ffill(inplace=True)

    return df_fac


@jitclass([
    ['max_leverage', nb.float64],  # 最大杠杆率 
    ['max_loss', nb.float64],  # 开仓最大亏损
    ['in_stop', nb.boolean],  # 是否在移动止损过程
    ['stop_pct', nb.float64],  # 移动止损比例
    ['stop_close_pct', nb.float64],  # 移动止损平仓比例
    ['face_value', nb.float64],  # 合约面值
    ['highest', nb.float64],  # 开仓后最高价
    ['lowest', nb.float64],  # 开仓后最低价
    ['prev_upper', nb.float64],  # 上根k线上轨
    ['prev_lower', nb.float64],  # 上根k线下轨
    ['prev_median', nb.float64],  # 上根k线均线
    ['prev_close', nb.float64]  # 上根k线收盘价
])
class Strategy:

    def __init__(self, max_leverage, max_loss, face_value, stop_pct, stop_close_pct):
        self.max_leverage = max_leverage
        self.max_loss = max_loss
        self.stop_pct = stop_pct
        self.stop_close_pct = stop_close_pct
        self.face_value = face_value

        self.in_stop = False
        self.highest = np.nan
        self.lowest = np.nan

        self.prev_upper = np.nan
        self.prev_lower = np.nan
        self.prev_median = np.nan
        self.prev_close = np.nan

    # 重置止损高低价
    def reset_stop(self, price):
        self.in_stop = True
        self.highest = price
        self.lowest = price

    # 更新止损高低价
    def update_stop_hl(self, price):
        if not self.in_stop:
            return
        self.highest = max(price, self.highest)
        self.lowest = min(price, self.lowest)

    def on_bar(self, candle, factors, pos, equity):
        op, hi, lo, cl, vol = candle
        upper, lower, median = factors

        # 默认保持原有仓位
        target_pos = pos

        # 移动止损中，更新高低价
        self.update_stop_hl(cl)

        if not np.isnan(self.prev_close):
            # 先判断开仓
            # 计算本次使用的杠杆
            risk = abs(cl / median - 1) + 1e-8
            leverage = min(self.max_loss / risk, self.max_leverage)

            # 做空或无仓位，上穿上轨，做多
            if pos <= 0 and cl > upper and self.prev_close <= self.prev_upper:
                target_pos = int(equity * leverage / cl / self.face_value)
                # 用当前价重置止损
                self.reset_stop(cl)

            # 做多或无仓位，下穿下轨，做空
            elif pos >= 0 and cl < lower and self.prev_close >= self.prev_lower:
                target_pos = -int(equity * leverage / cl / self.face_value)
                # 用当前价重置止损
                self.reset_stop(cl)

            # 目前持有做多仓位
            elif pos > 0:
                # 下穿中轨，平仓
                if cl < median and self.prev_close >= self.prev_median:
                    target_pos = 0

                # 移动止损过程中，跌了超过 self.stop_pct
                elif self.in_stop and (1 - cl / self.highest) > self.stop_pct:
                    # 平 self.stop_close_pct 的仓位
                    target_pos = int(pos * (1 - self.stop_close_pct))
                    self.reset_stop(cl)

            # 目前持有做空仓位
            elif pos < 0:
                #上穿中轨，平仓
                if cl > median and self.prev_close <= self.prev_median:
                    target_pos = 0

                # 移动止损过程中，涨了超过 self.stop_pct
                elif self.in_stop and (cl / self.lowest - 1) > self.stop_pct:
                    # 平 self.stop_close_pct 的仓位
                    target_pos = int(pos * (1 - self.stop_close_pct))
                    self.reset_stop(cl)

        # 更新上根K线数据
        self.prev_upper = upper
        self.prev_lower = lower
        self.prev_close = cl
        self.prev_median = median

        return target_pos


def get_default_factor_params_list():
    params = []
    for interval in ['1h', '30m']:  # 长周期
        for n in range(10, 101, 10):  # 均线周期
            for b in [1.5, 1.8, 2, 2.2, 2.5]:  # 布林带宽度
                params.append({'itl': interval, 'n': n, 'b': b})
    return params
