from twitter_data import tw_symbols
import requests
import tokens


model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
hf_token = tokens.HF_ACCESS_TOKEN

API_URL = "https://api-inference.huggingface.co/models/" + model
headers = {"Authorization": "Bearer %s" % (hf_token)}

def analysis(data):
    payload = dict(inputs=data, options=dict(wait_for_model=True))
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

print("for loop in")
tweets_analysis = []
for tweet in tw_symbols['tw_btc']['text'].to_numpy():
    try:
        sentiment_result = analysis(tweet)
        print("sentiment_result: ",sentiment_result)
        top_sentiment = max(sentiment_result, key=lambda x: x['score']) # Get the sentiment with the higher score
        print("top_sentiment: ",top_sentiment)
        tweets_analysis.append({'tweet': tweet, 'sentiment': top_sentiment['label']})
 
    except Exception as e:
        print("Sentiment analysis failed: ", e)
print("for loop out")
        
pd.set_option('max_colwidth', None)
pd.set_option('display.width', 3000)
tw_btc_df = pd.DataFrame(tweets_analysis)
 
# Show a tweet for each sentiment
display(tw_btc_df[tw_btc_df["sentiment"] == 'Positive'].head(1))
display(tw_btc_df[tw_btc_df["sentiment"] == 'Neutral'].head(1))
display(tw_btc_df[tw_btc_df["sentiment"] == 'Negative'].head(1))

sentiment_counts = tw_btc_df.groupby(['sentiment']).size()
print(sentiment_counts)