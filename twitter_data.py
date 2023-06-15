import pandas as pd
import json
from pandas import json_normalize

from datetime import datetime
import os

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"] 

keywords_list = ["bitcoin OR btc", "ethereum OR eth", "binance OR bnb", "ripple OR xrp", "cardano OR ada"]  #remember to update! 12.06
#start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
#end_date = datetime.strptime("2021-03-01", "%Y-%m-%d")

def create_tw_df(from_date):
    tw_symbols = {}
    
    for i, s in enumerate(symbols):
        coin = s[0: 3].lower()
        data_frames = []
        
        for date in from_dates:
            directory = "data/twitter"
            filename = f"{keywords_list[i]}_{date}.json"
            file_path = os.path.join(directory, filename)
            
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        if 'data' in item['data'] and 'items' in item['data']['data']:
                            df = pd.DataFrame.from_dict(item['data']['data']['items'])
                            data_frames.append(df)
        
        tw_symbols[f"tw_{coin}"] = pd.concat(data_frames)
        
    return tw_symbols

   
from_dates = ["11.22", "01.23"]
tw_symbols = create_tw_df(from_dates)

print(tw_symbols['tw_btc']['created_time'])
