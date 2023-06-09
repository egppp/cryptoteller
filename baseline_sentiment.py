from data_source import btc_data, eth_data, bnb_data, xrp_data, ada_data

df = [btc_data, eth_data, bnb_data, xrp_data, ada_data]
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

symbols_df = {}
for i, s in enumerate(symbols):
    coin = (s[0: 3].lower())
    symbols_df[f'{coin}'] = df[i].copy()
    symbols_df[f'{coin}'] = symbols_df[f'{coin}'][['close_time', 'close']]
    