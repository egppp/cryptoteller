from twitter_data import tw_symbols
import requests
import tokens
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

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

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

sentiment_df = {}
for i, s in enumerate(symbols):
    coin = (s[0: 3].lower())
    sentiment_df[f"sent_{coin}"] = tw_symbols[f"tw_{coin}"][['id', 'text', 'timestamp']].copy()
    
    sentiment_df[f'sent_{coin}']['timestamp'] = pd.to_datetime(sentiment_df[f'sent_{coin}']['timestamp'], unit = 'ms')

    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].text.apply(func = emoji)
    #sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = remove_users)
    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = clean_html)
    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = remove_links)
    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = non_ascii)
    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = lower)
    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = email_address)
    #sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = removeStopWords)
    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = clean_html)
    sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = punct)
    #sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = remove_)

#Sentiment Classification

    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest',from_tf=True)

    def sentiment_score(review):
        tokens = tokenizer.encode(review, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits))

    sentiment_df[f"sent_{coin}"]['sentiment'] = sentiment_df[f"sent_{coin}"]['new_tweet'].apply(lambda x: sentiment_score(x[:512]))

    print(sentiment_df[f"sent_{coin}"].head())
