import pandas as pd

def clean(df):
    # Create a copy of the DataFrame
    clean_df = df.copy()

    # Change the time unit of the "open_time" and "close_time" columns
    clean_df["open_time"] = pd.to_datetime(clean_df["open_time"], unit='ms', origin='unix')
    clean_df["close_time"] = pd.to_datetime(clean_df["close_time"], unit='ms', origin='unix')

    # Set the "open_time" column as the index
    clean_df = clean_df.set_index('open_time')

    # Drop the "high", "low", and "ignore" columns
    clean_df = clean_df.drop(columns=['high', 'low', 'ignore'])

    # Sort DataFrame by index in ascending order
    clean_df = clean_df.sort_index(ascending=True)

    # Return the cleaned DataFrame
    return clean_df
