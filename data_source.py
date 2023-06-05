import pandas as pd
from binance.client import Client
import datetime as dt
# client configuration
api_key = 'MY_API' 
api_secret = 'MY_SECRET_API'
client = Client(api_key, api_secret)

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
interval = '1d'

now = dt.datetime.now(dt.timezone.utc)
start = now - dt.timedelta(days=398) #30apr22 as of 2jun23
end = now - dt.timedelta(days=214) #31oct22 as of 2jun23

# Gives you a timestamp in ms
start_str = int(round(start.timestamp() * 1000, 0))
end_str = int(round(end.timestamp() * 1000, 0))

def get_data():
for symbol in symbols:
    Client.KLINE_INTERVAL_1DAY 
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    data = pd.DataFrame(klines)
    # create colums name
    data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol','ignore']
                
    # change the timestamp
    data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
    data.to_csv(symbol+'.csv', index = None, header=True)
    return "data sourced for {symbols}"


data=data.astype(float)
data["close"].plot(title = 'DOTUSDT', legend = 'close')