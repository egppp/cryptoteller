def split_data(df, train_ratio=0.8):
    # Calculate the index to split the data
    split_index = int(len(df) * train_ratio)
    
    # Split the data into training and test sets
    train_data = df[:split_index].copy()
    test_data = df[split_index:].copy()
    
    return train_data, test_data


from statsmodels.tsa.stattools import adfuller

def check_stationarity(df):
    # Perform ADF test on the "close" column
    result = adfuller(df_train["close"])
    
    # Extract the p-value from the test result
    p_value = result[1]
    
    # Check if the p-value is less than a significance level (e.g., 0.05)
    if p_value < 0.05:
        print("The 'close' column is stationary.")
    else:
        print("The 'close' column is not stationary.")



