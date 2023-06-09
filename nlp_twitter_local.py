from twitter_data import tw_symbols
import requests
import tokens
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


#Data Preprocessing

# translate emoji
def emoji(text):
  for emot in UNICODE_EMOJI:
    if text == None:
      text = text
    else:
      text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
    return text

def remove_links(tweet):
  '''Takes a string and removes web links from it'''
  tweet = re.sub(r'http\S+', '', tweet) # remove http links
  tweet = re.sub(r'bit.ly/\S+', '', tweet) # remove bitly links
  tweet = tweet.strip('[link]') # remove [links]
  return tweet

def clean_html(text):
  html = re.compile('<.*?>')#regex
  return html.sub(r'',text)

def non_ascii(s):
  return "".join(i for i in s if ord(i)<128)

def lower(text):
  return text.lower()

def email_address(text):
  email = re.compile(r'[\w\.-]+@[\w\.-]+')
  return email.sub(r'',text)

def punct(text):
  token=RegexpTokenizer(r'\w+')#regex
  text = token.tokenize(text)
  text= " ".join(text)
  return text

tw_btc_df = tw_symbols['tw_btc'][['id', 'text', 'timestamp']]

tw_btc_df['new_tweet'] = tw_btc_df.text.apply(func = emoji)
#tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = remove_users)
tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = clean_html)
tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = remove_links)
tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = non_ascii)
tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = lower)
tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = email_address)
#tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = removeStopWords)
tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = clean_html)
tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = punct)
#tw_btc_df['new_tweet'] = tw_btc_df.new_tweet.apply(func = remove_)

#Sentiment Classification

tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest',from_tf=True)

def sentiment_score(review):
  tokens = tokenizer.encode(review, return_tensors='pt')
  result = model(tokens)
  return int(torch.argmax(result.logits))

tw_btc_df['sentiment'] = tw_btc_df['new_tweet'].apply(lambda x: sentiment_score(x[:512]))

print(tw_btc_df.head())

'''
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
'''