# NEO 系统：一个高效的 K 线级在线趋势追踪回测系统

Numba-based Evaluation Online System, 缩写为 NEO System, 取救世主 NEO 之名

NEO 系统使用了基于 Numba 的 jit 编译，同时保证了回测的灵活和高效性

## 使用 NEO 系统回测一个布林带突破策略

Jupyter notebook [在这里](https://github.com/lostleaf/neo_trend_backtesting/blob/master/strategy_example/boll.ipynb)

`BollingStrategy` 策略实现[在这里](https://github.com/lostleaf/neo_trend_backtesting/blob/master/strategy_example/bolling.py)

首先，定义参数

``` python
# 回测起始日
START_DATE = '20180301'

# 布林带周期与宽度
N = 100
B = 2

# 回测参数
INIT_CAPITAL = 1e5  # 初始资金，10万
LEVERAGE = 1  # 使用1倍杠杆
FACE_VALUE = 0.01  # 合约面值 0.01
COMM_RATE = 6e-4  # 交易成本万分之 6
LIQUI_RATE = 5e-3  # 爆仓保证金率千分之 5
```

然后，读入 K 线行情并计算技术指标

`df` 是一个 pandas DataFrame，包含了 时间戳 `candle_begin_time` 和 K 线 OHLCV

``` python
df = pd.read_feather('../candle_1h.fea')

upper, median, lower = ta.BBANDS(
    df['close'], 
    timeperiod=N, nbdevup=B, nbdevdn=B, matype=ta.MA_Type.SMA)

df['upper'] = upper
df['median'] = median
df['lower'] = lower

df = df[df['candle_begin_time'] >= START_DATE].reset_index(drop=True)
```

最后，使用 NEO 系统回测

``` python
# 初始化回测模拟器
simulator = SingleFutureSimulator(
    init_capital=INIT_CAPITAL, 
    face_value=FACE_VALUE, 
    comm_rate=COMM_RATE, 
    liqui_rate=LIQUI_RATE, 
    init_pos=0)

# 初始化策略
strategy = BollingStrategy(leverage=LEVERAGE, face_value=FACE_VALUE)

# 运行回测
backtest_online(
    df, 
    simulator=simulator,
    strategy=strategy,
    factor_columns = ['upper', 'lower', 'median'])

# 回测结果
print(df[['candle_begin_time', 'pos', 'equity']].tail().to_markdown(), '\n')
```

|       | candle_begin_time   |     pos |      equity |
|------:|:--------------------|--------:|------------:|
| 37527 | 2022-06-13 19:00:00 | -321335 | 6.859e+06   |
| 37528 | 2022-06-13 20:00:00 | -321335 | 6.81568e+06 |
| 37529 | 2022-06-13 21:00:00 | -321335 | 6.85607e+06 |
| 37530 | 2022-06-13 22:00:00 | -321335 | 6.97178e+06 |
| 37531 | 2022-06-13 23:00:00 | -321335 | 6.91963e+06 | 
