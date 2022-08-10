# NEO System: An Online Candlestick based Trend Following Backtesting System

Numba-based Evaluation Online System, short for NEO System, named after *The One* Neo from *The Matrix*

## Backtest a Bollinger Band breakout strategy

The correlated jupyter notebook is here

First, specify the parameters

``` python
START_DATE = '20180301'

# Bollinger Band period and width
N = 100
B = 2

# Backtesting parameters
INIT_CAPITAL = 1e5  # Initial capital, 100K USD
LEVERAGE = 1  # 100% leverage
FACE_VALUE = 0.01  # Face value of each contract is 0.01
COMM_RATE = 6e-4  # Trading cost 0.06%
LIQUI_RATE = 5e-3  # Maintenance Margin rate 0.5%, will liquidate if lower
```

Then, load the candlesticks and compute the factors

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

Finally, backtest with NEO System

``` python
%%time

# Initalize the backtesting simulator
simulator = SingleFutureSimulator(
    init_capital=INIT_CAPITAL, 
    face_value=FACE_VALUE, 
    comm_rate=COMM_RATE, 
    liqui_rate=LIQUI_RATE, 
    init_pos=0)

# Initalize the Strategy
strategy = BollingStrategy(leverage=LEVERAGE, face_value=FACE_VALUE)

# Run backtesting
backtest_online(
    df, 
    simulator=simulator,
    strategy=strategy,
    factor_columns = ['upper', 'lower', 'median'])

# Print out the results
print(df[['candle_begin_time', 'pos', 'equity']].tail().to_markdown(), '\n')
```

|       | candle_begin_time   |     pos |      equity |
|------:|:--------------------|--------:|------------:|
| 37527 | 2022-06-13 19:00:00 | -321335 | 6.859e+06   |
| 37528 | 2022-06-13 20:00:00 | -321335 | 6.81568e+06 |
| 37529 | 2022-06-13 21:00:00 | -321335 | 6.85607e+06 |
| 37530 | 2022-06-13 22:00:00 | -321335 | 6.97178e+06 |
| 37531 | 2022-06-13 23:00:00 | -321335 | 6.91963e+06 | 
