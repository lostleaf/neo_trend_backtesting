import os
from datetime import timedelta

import pandas as pd


def to_timedelta(interval):
    if interval[-1] == 'm' or interval[-1] == 'T':
        return timedelta(minutes=int(interval[:-1]))
    if interval[-1] == 'h' or interval[-1] == 'H':
        return timedelta(hours=int(interval[:-1]))
    raise ValueError(f'Unknown time interval {interval}')


def read_candle_feather(path):
    df = pd.read_feather(path)
    if 'candle_end_time' not in df:
        interval = os.path.basename(path).split('.')[0].split('_')[1]
        delta = to_timedelta(interval)
        df['candle_end_time'] = df['candle_begin_time'] + delta

    # OHLCV with timestamps
    return df[['candle_begin_time', 'candle_end_time', 'open', 'high', 'low', 'close', 'volume']]