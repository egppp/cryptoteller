import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential


url = "/home/yass/code/yass2803/cryptoteller/data/ADAUSDT.csv"
df = pd.read_csv(url)
df.head()


# Change the time unit of the "open_time" and "close_time" columns
df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', origin='unix')
df = df.drop(columns=["close_time"])


#Plot the Historical Price
ax = df.plot(x='open_time', y='close');
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
plt.show()



# Normalization
scaler = MinMaxScaler()

close_price = df.close.values.reshape(-1, 1)

scaled_close = scaler.fit_transform(close_price)


# Preprocessing
class Preprocessor:
    def __init__(self, SEQ_LEN=100):
        self.SEQ_LEN=SEQ_LEN
        pass
    def to_sequences(self, data, seq_len):
        d = []

        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])

        return np.array(d)

    def preprocess(self, data_raw, train_split):

        data = self.to_sequences(data_raw, self.SEQ_LEN)

        self.num_train = int(train_split * data.shape[0])

        X_train = data[:self.num_train, :-1, :]
        y_train = data[:self.num_train, -1, :]

        X_test = data[self.num_train:, :-1, :]
        y_test = data[self.num_train:, -1, :]

        return X_train, y_train, X_test, y_test

preprocessor=Preprocessor()
X_train, y_train, X_test, y_test = preprocessor.preprocess(scaled_close, train_split = 0.80)


# Model

DROPOUT = 0.2
WINDOW_SIZE = preprocessor.SEQ_LEN - 1

model = keras.Sequential()

model.add(LSTM(WINDOW_SIZE, return_sequences=True, input_shape=(WINDOW_SIZE, X_train.shape[-1])),
                        )
model.add(Dropout(rate=DROPOUT))

model.add(LSTM(WINDOW_SIZE * 2, return_sequences=True))
model.add(Dropout(rate=DROPOUT))

model.add(LSTM(WINDOW_SIZE, return_sequences=False))

model.add(Dense(units=1))

model.add(Activation('linear'))


# Training


from tensorflow import keras
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor="val_loss",
                       patience=3,
                       mode="min",
                       restore_best_weights=True)

model.compile(
    loss='mean_squared_error',
    metrics=["mae"] ,
    optimizer='adam'
)



BATCH_SIZE = 64

history = model.fit(
    X_train,
    y_train,
    epochs=6,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=0.1,
    callbacks=[es]
)




MAE=scaler.inverse_transform([[model.evaluate(X_test, y_test)[1]]])[0][0]
print(f"MAE for the test set: {MAE} Dollars")



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Prediction Plot
# Prediction
y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

date_time_test = df.iloc[preprocessor.num_train + preprocessor.SEQ_LEN:, 0].to_numpy()

# Plot the actual and predicted prices
plt.figure(figsize=(12, 8))  # Set the figure size to make it larger
plt.plot(date_time_test, y_test_inverse, label="Actual Price", color='green')
plt.plot(date_time_test, y_hat_inverse, label="Predicted Price", color='red')

# Customize the plot
plt.title('ADA Price Prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees for better readability
plt.legend(loc='best')

plt.show()
