import numba as nb
import numpy as np
import pandas as pd
from numba.experimental import jitclass


def boll_factor(df: pd.DataFrame, n, b):
    """
    单周期的布林带因子
    """
    # 计算均线
    df['median'] = df['close'].rolling(n, min_periods=1).mean()

    # 计算标准差
    std = df['close'].rolling(n, min_periods=1).std(ddof=0)  # ddof代表标准差自由度

    # 计算上轨、下轨道
    df['upper'] = df['median'] + b * std
    df['lower'] = df['median'] - b * std

    return df


def boll_factor_cross_timeframe(df_1m: pd.DataFrame, df_1h: pd.DataFrame, n, b):
    """
    跨周期的布林带因子
    先在1小时线（大周期）上计算布林带，然后填充到1分钟（小周期）上
    每根1分钟线均使用最近的已闭合的1小时线上中下轨
    """
    df_1h = boll_factor(df_1h, n, b)

    # 根据 k 线结束时间对齐
    df_1h['candle_end_time'] = df_1h['candle_begin_time'] + pd.Timedelta(hours=1)
    df_1m['candle_end_time'] = df_1m['candle_begin_time'] + pd.Timedelta(minutes=1)

    factor_cols = ['upper', 'lower', 'median']
    df = df_1m.join(df_1h.set_index('candle_end_time')[factor_cols], on='candle_end_time')

    # 向后填充分钟线上中下轨
    for col in factor_cols:
        df[col].ffill(inplace=True)

    return df


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
class BollingPosMgtStrategy:

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
