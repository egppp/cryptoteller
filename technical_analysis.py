from data_source import btc_data, eth_data, bnb_data, xrp_data, ada_data

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

import ta
import talib as tal
import pandas_ta as pta
from finta import TA
from ta.momentum import StochasticOscillator

df = [btc_data, eth_data, bnb_data, xrp_data, ada_data]
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
ta_symbols = {}
for i, s in enumerate(symbols):
    coin = (s[0: 3].lower())
    ta_symbols[f'ta_{coin}'] = df[i].copy()
    
    ta_symbols[f'ta_{coin}']['open_time'] = pd.to_datetime(ta_symbols[f'ta_{coin}']['open_time'], unit = 'ms')
    
    #MOMENTUM INDICATORS
    #RSI - Relative Strength Index
    ta_symbols[f'ta_{coin}']['RSI'] = tal.RSI(ta_symbols[f'ta_{coin}']['close'], timeperiod=14)

    #Stochastic Oscillator
    ta_symbols[f'ta_{coin}']['%K'] = ta.momentum.stoch(ta_symbols[f'ta_{coin}']['high'], ta_symbols[f'ta_{coin}']['low'], ta_symbols[f'ta_{coin}']['close'] , 14 ,3)
    ta_symbols[f'ta_{coin}']['%D'] = ta_symbols[f'ta_{coin}']['%K'].rolling(3).mean()

    #MACD - Moving Average Convergence Divergence
    ta_symbols[f'ta_{coin}'].ta.macd(close=ta_symbols[f'ta_{coin}']['close'], fast=12, slow=26, signal=9, append=True)

    #Ichimoku Cloud
    # Define length of Tenkan Sen or Conversion Line
    cl_period = 20 

    # Define length of Kijun Sen or Base Line
    bl_period = 60  

    # Define length of Senkou Sen B or Leading Span B
    lead_span_b_period = 120  

    # Define length of Chikou Span or Lagging Span
    lag_span_period = 30  

    # Calculate conversion line
    high_20 = ta_symbols[f'ta_{coin}']['high'].rolling(cl_period).max()
    low_20 = ta_symbols[f'ta_{coin}']['low'].rolling(cl_period).min()
    ta_symbols[f'ta_{coin}']['conversion_line'] = (high_20 + low_20) / 2

    # Calculate based line
    high_60 = ta_symbols[f'ta_{coin}']['high'].rolling(bl_period).max()
    low_60 = ta_symbols[f'ta_{coin}']['low'].rolling(bl_period).min()
    ta_symbols[f'ta_{coin}']['base_line'] = (high_60 + low_60) / 2

    # Calculate leading span A
    ta_symbols[f'ta_{coin}']['lead_span_A'] = ((ta_symbols[f'ta_{coin}'].conversion_line + ta_symbols[f'ta_{coin}'].base_line) / 2).shift(lag_span_period)

    # Calculate leading span B
    high_120 = ta_symbols[f'ta_{coin}']['high'].rolling(120).max()
    low_120 = ta_symbols[f'ta_{coin}']['high'].rolling(120).min()
    ta_symbols[f'ta_{coin}']['lead_span_B'] = ((high_120 + low_120) / 2).shift(lead_span_b_period)

    # Calculate lagging span
    ta_symbols[f'ta_{coin}']['lagging_span'] = ta_symbols[f'ta_{coin}']['close'].shift(-lag_span_period)

    #CCI-Commodity Channel Index
    ta_symbols[f'ta_{coin}']['CCI'] = tal.CCI(ta_symbols[f'ta_{coin}']['high'], ta_symbols[f'ta_{coin}']['low'], ta_symbols[f'ta_{coin}']['close'], timeperiod=14)

    print(ta_symbols[f'ta_{coin}'].head())

    ta_symbols[f'ta_{coin}'].to_csv(f'ta_{coin}.csv')
    