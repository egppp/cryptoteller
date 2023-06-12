from data_source import btc_data, eth_data, bnb_data, xrp_data, ada_data
from nlp_twitter_local import sentiment_df

df = [btc_data, eth_data, bnb_data, xrp_data, ada_data]
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

symbols_df = {}
for i, s in enumerate(symbols):
    coin = (s[0: 3].lower())
    symbols_df[f'{coin}'] = df[i].copy()
    symbols_df[f'{coin}'] = symbols_df[f'{coin}'][['close_time', 'close']]
    symbols_df[f'{coin}']['close_time'] = pd.to_datetime(symbols_df[f'ta_{coin}']['close_time'], unit = 'ms')
    symbols_df[f'{coin}'].merge(sentiment_df[f"sent_{coin}"], left_index=True, right_index=True)
    
    print(symbols_df[f'{coin}'].head())