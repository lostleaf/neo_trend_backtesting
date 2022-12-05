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
    df_fac = pd.DataFrame({
        'upper': upper,
        'median': median,
        'lower': lower,
        'candle_end_time': df_long['candle_end_time']
    })

    # 填充到1分钟
    df_fac.set_index('candle_end_time', inplace=True)
    df_fac = df_1m.join(df_fac, on='candle_end_time')
    for col in FCOLS:
        df_fac[col].ffill(inplace=True)

    return df_fac


@jitclass([
    ['leverage', nb.float64],  # 杠杆率 
    ['face_value', nb.float64],  # 合约面值
    ['prev_upper', nb.float64],  # 上根k线上轨
    ['prev_lower', nb.float64],  # 上根k线下轨
    ['prev_median', nb.float64],  # 上根k线均线
    ['prev_close', nb.float64]  # 上根k线收盘价
])
class Strategy:

    def __init__(self, leverage, face_value):
        self.leverage = leverage
        self.face_value = face_value

        self.prev_upper = np.nan
        self.prev_lower = np.nan
        self.prev_median = np.nan
        self.prev_close = np.nan

    def on_bar(self, candle, factors, pos, equity):
        cl = candle['close']
        upper = factors['upper']
        lower = factors['lower']
        median = factors['median']

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


def get_default_factor_params_list():
    params = []
    for interval in ['1h', '30m']:  # 长周期
        for n in range(10, 101, 10):  # 均线周期
            for b in [1.5, 1.8, 2, 2.2, 2.5]:  # 布林带宽度
                params.append({'itl': interval, 'n': n, 'b': b})
    return params


def get_default_strategy_params_list():
    params = []
    for lev in [1, 1.5]:
        params.append({'leverage': lev})

    return params