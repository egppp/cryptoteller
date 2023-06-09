import pandas as pd
import json
from pandas import json_normalize
import requests


import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import glob

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
keywords_list = ["bitcoin", "ethereum", "binance", "ripple", "cardano"]


#REQUEST
base_url = "https://api.data365.co/v1.1/twitter/search/post/update"
get_url = "https://api.data365.co/v1.1/twitter/search/post/posts"
access_token = 'ZXlKMGVYQWlPaUpLVjFRaUxDSmhiR2NpT2lKSVV6STFOaUo5LmV5SnpkV0lpT2lKRmJHbDZZV0psZEdoSGFYSmhiR1J2SWl3aWFXRjBJam94TmpnMk1EY3pNalU0TGpJM09UWXlOSDAuVDJsSEp6T0IyRjRyalNEcXNGdWRXV3hqUF80VjUtLW9iYzdNM21GOTZOZw=='

keywords_list = ["bitcoin OR btc", "ethereum OR eth", "binance OR bnb", "ripple OR xrp", "cardano OR ada"]  # Example list of keywords
start_date = datetime.strptime("2020-01-15", "%Y-%m-%d")
end_date = datetime.strptime("2020-03-10", "%Y-%m-%d")

for keyword in keywords_list:
    keyword_data = []

    current_date = start_date

    while current_date < end_date:
        from_date = current_date.strftime("%Y-%m-%d")
        to_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Increase by 1 day

        params = {
            "keywords": keyword,
            "search_type": "top",
            "max_posts": 30,
            "request": "crypto OR cryptocurrencies OR cryptocurrency",
            "lang": "en",
            "from_date": from_date,
            "to_date": to_date,
            "access_token": access_token
        }

        post_response = requests.post(base_url, params=params)

        print(post_response.text)  # Print the response for each iteration
        print("Status code: ", post_response.status_code) 
        
        if post_response.status_code == 202:
            print("POST request successful")

            # Extract data from POST response
            post_data = post_response.json()
            # ...
            time.sleep(130)
            # Make GET request for search results
            get_params = {
                "lang": "en",
                "keywords": keyword,
                "search_type": "latest",
                "request": "crypto OR cryptocurrencies OR cryptocurrency",
                "from_date": from_date,
                "to_date": to_date,
                "access_token": access_token 
            }

            get_response = requests.get(get_url, params=get_params)

            if get_response.status_code == 200:
                print("GET request successful")
                # Extract data from GET response
                get_data = get_response.json()
                
                keyword_data.append({
                    "from_date": from_date,
                    "to_date": to_date,
                    "data": get_data
                })

            else:
                print(f"GET request failed with status code: {get_response.status_code}")
                # Handle GET request error

        else:
            print(f"POST request failed with status code: {post_response.status_code}")
            # Handle POST request error

        current_date += timedelta(days=1)
    
    # Save keyword data as JSON file
    file_name = f"{keyword}.json"
    with open(file_name, "w") as file:
        json.dump(keyword_data, file)
    print(f"Saved response for {keyword} as {file_name}")


'''
def create_tw_df():
    tw_symbols = {}    
    for i, s in enumerate(symbols):
        coin = s[0: 3].lower()
        data_frames = []
        # Specify the directory path
        directory = "data/twitter"
        # Use glob to find files starting with "bitcoin" in the directory
        fpath = f"{directory}/{keywords_list[i]}*"
        files = glob.glob(fpath)
        
        # Iterate through the files using a for loop
        
        for file in files:     
            with open(file, "r") as f:
                data = json.load(f)
                if 'data' in data and 'items' in data['data']:
                    df = pd.DataFrame.from_dict(data['data']['items'])
                    data_frames.append(df)
 
        tw_symbols[f"tw_{coin}"] = pd.concat(data_frames)
        # print(tw_symbols[f"tw_{coin}"].info())
        
    return tw_symbols
        
tw_symbols = create_tw_df()        

print(tw_symbols['tw_eth']['text'])
'''