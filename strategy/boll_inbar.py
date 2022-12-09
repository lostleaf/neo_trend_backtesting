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
    ['prev_upper', nb.float64],  # 上根k线上轨
    ['prev_lower', nb.float64],  # 上根k线下轨
    ['prev_median', nb.float64],  # 上根k线均线
    ['prev_close', nb.float64],  # 上根k线收盘价
    ['exposure', nb.float64],  # 风险敞口
    ['leverage', nb.float64]
])
class Exposure:

    def __init__(self, leverage):
        self.exposure = 0
        self.leverage = leverage

        self.prev_upper = np.nan
        self.prev_lower = np.nan
        self.prev_median = np.nan
        self.prev_close = np.nan

    def on_bar(self, candle, factors, equity_usd):
        cl = candle['close']
        upper = factors['upper']
        lower = factors['lower']
        median = factors['median']

        if not np.isnan(self.prev_close):
            # 做空或无仓位，上穿上轨，做多
            if self.exposure <= 0 and cl > upper and self.prev_close <= self.prev_upper:
                self.exposure = equity_usd * self.leverage / cl

            # 做多或无仓位，下穿下轨，做空
            elif self.exposure >= 0 and cl < lower and self.prev_close >= self.prev_lower:
                self.exposure = -equity_usd * self.leverage / cl

            # 做多，下穿中轨，平仓
            elif self.exposure > 0 and cl < median and self.prev_close >= self.prev_median:
                self.exposure = 0

            # 做空，上穿中轨，平仓
            elif self.exposure < 0 and cl > median and self.prev_close <= self.prev_median:
                self.exposure = 0

        # 更新上根K线数据
        self.prev_upper = upper
        self.prev_lower = lower
        self.prev_close = cl
        self.prev_median = median

        return self.exposure


@jitclass
class Strategy:
    stra_pos: Exposure
    face_value: float
    leverage: float
    pre_exposure: float

    def __init__(self, leverage, face_value):
        self.stra_pos = Exposure(leverage)
        self.face_value = face_value
        self.pre_exposure = np.nan

    def on_bar(self, candle, factors, pos, equity):
        if np.isnan(self.pre_exposure):
            self.pre_exposure = pos * self.face_value

        target_exposure = self.stra_pos.on_bar(candle, factors, equity)

        # 风险敞口变化
        if self.pre_exposure != target_exposure:
            pos = int(target_exposure / self.face_value)
            self.pre_exposure = target_exposure
        return pos


@jitclass
class InverseStrategy:
    stra_pos: Exposure
    face_value: float
    leverage: float
    pre_exposure: float

    def __init__(self, leverage, face_value):
        self.stra_pos = Exposure(leverage)
        self.face_value = face_value
        self.pre_exposure = np.nan

    def on_bar(self, candle, factors, pos, equity):
        cl = candle['close']
        eq_usd = equity * cl

        if np.isnan(self.pre_exposure):
            self.pre_exposure = equity + pos * self.face_value / cl

        target_exposure = self.stra_pos.on_bar(candle, factors, eq_usd)
        
        # 风险敞口变化
        if target_exposure != self.pre_exposure:
            pos = int((target_exposure - equity) * cl / self.face_value)
            self.pre_exposure = target_exposure

        return pos


def get_default_factor_params_list():
    params = []
    for interval in ['1h', '30m']:  # 长周期
        for n in range(10, 201, 10):  # 均线周期
            for b in [1.8, 2, 2.2]:  # 布林带宽度
                params.append({'itl': interval, 'n': n, 'b': b})
    return params


def get_default_strategy_params_list():
    params = []
    for lev in [1]:
        params.append({'leverage': lev})

    return params
