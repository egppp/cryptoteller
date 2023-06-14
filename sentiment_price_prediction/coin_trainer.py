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
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
#from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential, Model
import random
import sklearn
from tensorflow.keras.regularizers import L1L2
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import os
from joblib import dump, load

class Preprocessor:
    def __init__(self, SEQ_LEN=70): #remember to change it back to 70
        self.SEQ_LEN=SEQ_LEN
        pass
    def to_sequences(self, data,sentiment, seq_len):
        d = []
        s=[]

        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])
            s.append(sentiment[index:index+seq_len])

        return np.array(d), np.array(s)

    def preprocess(self, data_raw,sentiment, train_split):

        data, sentiment = self.to_sequences(data_raw, sentiment, self.SEQ_LEN)

        self.num_train = int(train_split * data.shape[0])

        X_train = data[:self.num_train, :-1, :]
        y_train = data[:self.num_train, -1, :]
        sentiment_train=sentiment[:self.num_train, :-1, :]

        X_test = data[self.num_train:, :-1, :]
        y_test = data[self.num_train:, -1, :]
        sentiment_test=sentiment[self.num_train:, :-1, :]

        return X_train, y_train, X_test, y_test, sentiment_train, sentiment_test

def date_to_string(date_):
    year = str(date_.year)
    month = str(date_.month)
    day = str(date_.day)

    if len(day) == 1:
        day = "0" + day

    if len(month) == 1:
        month = "0" + month

    return year + "-" + month + "-" + day

def c_trainer(coin, plotting=False):
    # Read the data from a CSV file
    this_files_path=os.getcwd()
    url_price = os.path.join(this_files_path, "data", f"{coin}USDT.csv")
    df = pd.read_csv(url_price)

    # Change the time unit of the "open_time" column
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', origin='unix')

    # Normalization
    scaler = StandardScaler()
    close_price = df.close.values.reshape(-1, 1) # reshapes the numpy array so it can be pugged in the fit_transform
    scaled_close = scaler.fit_transform(close_price)

    scaled_close.shape

    np.isnan(scaled_close).any()

    url_sentiment = os.path.join(this_files_path, "data", "sentiment", f"aggr_sent_{coin.lower()}.csv")
    df_sentiment = pd.read_csv(url_sentiment)

    df_dates = df[["open_time"]].copy()
    df_dates.loc[:,"date"] = df_dates.open_time.apply(lambda x: date_to_string(x))

    df_sentiment = df_dates.merge(df_sentiment, on="date", how="left")
    df_sentiment.drop(columns="date", inplace=True)

    not_nan_idx = np.where(~np.isnan(df_sentiment.weighted_avg.values))[0]

    sentiment = df_sentiment.iloc[not_nan_idx,:].weighted_avg.values.reshape(-1, 1)

    scaled_returns_red = scaled_close[not_nan_idx]

    #preprpcessing
    preprocessor=Preprocessor()
    X_train, y_train, X_test, y_test, sentiment_train, sentiment_test = preprocessor.preprocess(scaled_returns_red,sentiment, train_split = 0.80)
    X_train=np.concatenate([X_train,sentiment_train],axis=2)
    X_test=np.concatenate([X_test,sentiment_test],axis=2)

    print(f"sentiment train dtype: {sentiment_train.dtype}")

    # Model
    DROPOUT = 0.2
    WINDOW_SIZE = preprocessor.SEQ_LEN - 1


    model_seq = keras.Sequential()

    model_seq.add(LSTM(WINDOW_SIZE, return_sequences=True, input_shape=(WINDOW_SIZE, X_train.shape[-1])),
                            )
    model_seq.add(Dropout(rate=DROPOUT))

    model_seq.add(LSTM(WINDOW_SIZE * 2, return_sequences=False))
    model_seq.add(Dropout(rate=DROPOUT))

    #model_seq.add(LSTM(WINDOW_SIZE, return_sequences=False))

    model_seq.add(Dense(units=1))

    model_seq.add(Activation('linear'))



    # Training
    es = EarlyStopping(monitor="val_loss",
                        patience=10,
                        mode="min",
                        restore_best_weights=True)


    model_seq.compile(
        loss='mean_squared_error',
        metrics=["mae"] ,
        optimizer="adam"
    )

    BATCH_SIZE = 32

    history = model_seq.fit(
        X_train,
        y_train,
        epochs=40, #remember to change it to 40
        batch_size=BATCH_SIZE,
        shuffle=False,
        validation_split=0.1
    )

    model_seq.save(f'models/{coin}_model')
    dump(scaler, f'models/{coin}_scaler')
    np.save(f'models/{coin}_xtest.npy', X_test)
    
    # Evaluation
    MAE=scaler.inverse_transform([[model_seq.evaluate(X_test, y_test)[1]]])[0][0]
    print(f"MAE for the test set: {MAE}")
    
    if plotting == True:
        # Plot the training loss and validation loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{coin} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        # Prediction
        y_hat = model_seq.predict(X_test)
        # Inverse transform the scaled data to get the actual prices
        y_test_inverse = scaler.inverse_transform(y_test)
        y_hat_inverse = scaler.inverse_transform(y_hat)

        # Plot the actual and predicted prices
        # Prediction
        date_time_test = df.iloc[not_nan_idx,:].iloc[preprocessor.num_train + preprocessor.SEQ_LEN:, 0].to_numpy()

        # Plot the actual and predicted prices
        plt.figure(figsize=(12, 8))  # Set the figure size to make it larger
        plt.plot(date_time_test, y_test_inverse, label="Actual Price", color='green')
        plt.plot(date_time_test, y_hat_inverse, label="Predicted Price", color='red')

        # Customize the plot
        plt.title(f'{coin} Price Prediction')
        plt.xlabel('Time [days]')
        plt.ylabel('Price')
        plt.show()
        
        #Plot the closing price over time
        ax = df.plot(x='open_time', y='close');
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Close Price")
        ax.set_title(f"Evolution {coin} vs. USD")

        plt.show()
        
    
    return f"{coin} model is trained"

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
for s in symbols:
    coin = s[0: 3]
    print(c_trainer(coin, plotting=True))
    


