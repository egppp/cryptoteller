import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
#from tensorflow.python.keras.layers import CuDNNLSTM
#from tensorflow.keras.models import Sequential, Model
import random
import sklearn
from tensorflow.keras.regularizers import L1L2
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import os
from joblib import load


def c_predictor(coin, X_test):

    model = tf.keras.models.load_model(f'models/{coin}_model')
    scaler = load(f'models/{coin}_scaler')
    X_test = np.load(f'models/{coin}_xtest.npy')

    # Prediction
    y_hat = model.predict(X_test)
    # Inverse transform the scaled data to get the actual prices
    #y_test_inverse = scaler.inverse_transform(y_test)
    y_hat_inverse = scaler.inverse_transform(y_hat)
    np.save(f'models/{coin}_ypred.npy', y_hat_inverse)
    return y_hat_inverse

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
for s in symbols:
    coin = s[0: 3]
    X_test = np.load(f'models/{coin}_xtest.npy')
    print(c_predictor(coin, X_test))


