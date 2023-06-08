import pandas as pd
from binance.client import Client
import datetime as dt
# client configuration
api_key = 'MY_API' 
api_secret = 'MY_SECRET_API'


symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
interval = '6h'

now = dt.datetime.now(dt.timezone.utc)
start = dt.datetime(2020, 1, 1) 
end = dt.datetime(2023, 5, 31)  

# Gives you a timestamp in ms
start_str = int(round(start.timestamp() * 1000, 0))
end_str = int(round(end.timestamp() * 1000, 0))

def get_data():
    client = Client(api_key, api_secret)
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

#get_data()


btc_data = pd.read_csv("data/BTCUSDT.csv")
eth_data = pd.read_csv("data/ETHUSDT.csv")
bnb_data = pd.read_csv("data/BNBUSDT.csv")
xrp_data = pd.read_csv("data/XRPUSDT.csv")
ada_data = pd.read_csv("data/ADAUSDT.csv")

#data=data.astype(float)
#data["close"].plot(title = 'DOTUSDT', legend = 'close')

'''from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["bitcoin"] # list of keywords to get data 

pytrends.build_payload(kw_list, cat=0, timeframe='2022-01-01 2023-05-31')
gt_data = pytrends.interest_over_time()
gt_data.head()
'''
