from data_source import btc_data, eth_data, bnb_data, xrp_data, ada_data

import pandas as pd
import matplotlib.pyplot as plt

import talib as ta
import pandas_ta as pta
from finta import TA
from ta.momentum.StochasticOscillator

ta_btc = btc_data.copy()

#RSI - Relative Strength Index
ta_btc['rsi_real'] = ta.RSI(ta_btc['close'], timeperiod=14)
print(ta_btc.head())

#Stochastic Oscillator
ta_btc.ta.stoch(high='high', low='low', k=14, d=3, append=True)

