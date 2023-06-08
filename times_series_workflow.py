from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm


def check_stationarity(df):
    # Perform ADF test on the "close" column
    result = adfuller(df)

    # Extract the p-value from the test result
    p_value = result[1]

    # Check if the p-value is less than a significance level (e.g., 0.05)
    if p_value < 0.05:
        print("The 'close' column is stationary.")
    else:
        print("The 'close' column is not stationary.")




#remove_stationarity
    # Perform differencing
    differenced = df.diff().fillna(0)


# (def deseasonalize(df):
#     # Convertir el índice de tiempo a tipo DatetimeIndex con frecuencia
#     df = pd.to_datetime(df)

#     # Realizar la descomposición de la serie de tiempo
#     decomposition = sm.tsa.seasonal_decompose(df.asfreq(freq="D"), model='additive')

#     # Obtener la componente sin tendencia y estacionalidad
#     deseasonalized = df - decomposition.seasonal - decomposition.trend

#     # Agregar la columna desestacionalizada al DataFrame
#     df['deseasonalized'] = deseasonalized

#     return df
# )




# def linearize_dataframe(df):
#     # Apply logarithmic transformation to the "close" column
#     df["linearized"] = np.log(df["close"])

#     return df




def split_data(df, train_ratio=0.8):
    # Calculate the index to split the data
    split_index = int(len(df) * train_ratio)

    # Split the data into training and test sets
    train_data = df[:split_index].copy()
    test_data = df[split_index:].copy()

    return train_data, test_data


def fit_auto_arima(train_data):
    # Create the SARIMA model
    smodel = pm.auto_arima(train_data,
                           start_p=1, max_p=2,
                           start_q=1, max_q=2,
                           trend='t',
                           seasonal=False,
                           trace=True)

    return smodel
