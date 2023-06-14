import pandas as pd
import json
from pandas import json_normalize

from datetime import datetime
import glob

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"] 

keywords_list = ["bitcoin OR btc", "ethereum OR eth", "binance OR bnb", "ripple OR xrp", "cardano OR ada"]  #remember to update! 12.06
start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2021-03-01", "%Y-%m-%d")

def create_tw_df(from_date):
    tw_symbols = {}    
    for i, s in enumerate(symbols):
        coin = s[0: 3].lower()
        data_frames = []
        # Specify the directory path
        directory = "data/twitter"
        # Use glob to find files starting with "bitcoin" in the directory
        fpath = f"{directory}/{keywords_list[i]}_{from_date}*"
        files = glob.glob(fpath)
        print(files)
        
        for file in files: 
            with open(file, "r") as f:
                data = json.load(f)
                for i in range(len(data)):
                    if 'data' in data[i]['data'] and 'items' in data[i]['data']['data']:
                        df = pd.DataFrame.from_dict(data[i]['data']['data']['items'])
                        data_frames.append(df)
 
        tw_symbols[f"tw_{coin}"] = pd.concat(data_frames)
        
    return tw_symbols
   
tw_symbols = create_tw_df("06.21")

print(tw_symbols['tw_btc']['created_time'])
