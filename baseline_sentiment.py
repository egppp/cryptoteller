from data_source import btc_data, eth_data, bnb_data, xrp_data, ada_data
from nlp_twitter_local import Datagen
import os
import pandas as pd


this_files_path=os.getcwd()
sentiment_directory=os.path.join(this_files_path, "data", "sentiment")
'''
data_generator=Datagen(sentiment_directory)

generate=True
if generate:
    sentiment_df=data_generator.generate_sentiment()
else:
    sentiment_df=data_generator.read_csv()
'''
    

def aggregated_sentiment(csv_path, coin, output_path):
    # Read the CSV file
    sentiment_df = pd.read_csv(csv_path)
    # Perform sentiment aggregation
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], format='%Y-%m-%d %H:%M:%S').dt.date
    grouped_df = sentiment_df.groupby("timestamp")
    aggregated_sentiments = []
    
    for date, group in grouped_df:
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for score in group['sentiment']:
            if score == 0:
                neutral_count += 1
            elif score == 1:
                positive_count += 1
            elif score == 2:
                negative_count += 1

        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = 1  # positive
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = 2  # negative
        else:
            overall_sentiment = 0
            
        weighted_avg = (neutral_count * 0 + positive_count * 1 + negative_count * -1) / (neutral_count + positive_count + negative_count)

        aggregated_sentiments.append({"date": date, "sentiment": overall_sentiment, 
                                      "neutral_count": neutral_count, "positive_count": positive_count,
                                      "negative_count": negative_count, "weighted_avg": weighted_avg})
    
    agg_df = pd.DataFrame(aggregated_sentiments)
    agg_df.to_csv(os.path.join(output_path,f"aggr_sent_{coin}.csv"), index=False)
    return agg_df
    
    
df_prices = [btc_data, eth_data, bnb_data, xrp_data, ada_data]
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

agg_df = {}
for i, s in enumerate(symbols):
    coin = (s[0: 3].lower())
    csv_path=os.path.join(sentiment_directory, f'{coin}.csv')
    output_path = sentiment_directory
    agg_df[f'df_{coin}'] = aggregated_sentiment(csv_path, coin, output_path)

    '''
    symbols_df[f'df_{coin}'] = df_prices[i].copy()
    symbols_df[f'df_{coin}'] = symbols_df[f'df_{coin}'][['close_time', 'close']]
    #symbols_df[f'df_{coin}']['close_time'] = pd.to_datetime(symbols_df[f'df_{coin}']['close_time'], unit = 'ms')
    symbols_df[f'df_{coin}'].merge(sentiment_df[f"sent_{coin}"], left_index=True, right_index=True)
    
    print(sentiment_df[f"sent_{coin}"].iloc[500:510,3:6])
    '''