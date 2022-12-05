import os

import numpy as np
import pandas as pd

from .time import to_timedelta


def read_candle_feather(path):
    df = pd.read_feather(path)
    if 'candle_end_time' not in df:
        interval = os.path.basename(path).split('.')[0].split('_')[1]
        delta = to_timedelta(interval)
        df['candle_end_time'] = df['candle_begin_time'] + delta

    # OHLCV with timestamps
    return df[['candle_begin_time', 'candle_end_time', 'open', 'high', 'low', 'close', 'volume']]


def transform_np_struct(df: pd.DataFrame):
    arr = df.to_records(index=False)
    return arr.view(arr.dtype.fields or arr.dtype, np.ndarray)


def transform_candle_np_struct(df: pd.DataFrame):
    return transform_np_struct(df[['open', 'high', 'low', 'close', 'volume']])