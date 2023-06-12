from twitter_data import tw_symbols
import requests
import tokens
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os


#Data Preprocessing
class Datagen:
  def __init__(self,path,symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]):
    self.symbols=symbols
    self.path=path
    assert os.path.isdir(path)
    self.coins= [(s[0: 3].lower()) for s in symbols]
  # translate emoji
  def _emoji(self,text):
    for emot in UNICODE_EMOJI:
      if text == None:
        text = text
      else:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
      return text
    
  # remove retweet username and tweeted at @username
  def _remove_users(self,tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    # remove tweeted at
    return tweet

  def _remove_links(self,tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # remove bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

  def _clean_html(self,text):
    html = re.compile('<.*?>')#regex
    return html.sub(r'',text)

  def _non_ascii(self,s):
    return "".join(i for i in s if ord(i)<128)

  def _lower(self,text):
    return text.lower()

  def _email_address(self,text):
    email = re.compile(r'[\w\.-]+@[\w\.-]+')
    return email.sub(r'',text)
  
# remove stopwords
  def _removeStopWords(self,str):
  #select english stopwords
    cachedStopWords = set(stopwords.words("english"))
  #add custom words
    #cachedStopWords.update(('and','I','A','http','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','mailto','regards','ayanna','like','email'))
  #remove stop words
    new_str = ' '.join([word for word in str.split() if word not in cachedStopWords]) 
    return new_str

  def _punct(self,text):
    token=RegexpTokenizer(r'\w+')#regex
    text = token.tokenize(text)
    text= " ".join(text)
    return text

  def generate_sentiment(self):
    sentiment_df = {}
    for coin in self.coins:
        sentiment_df[f"sent_{coin}"] = tw_symbols[f"tw_{coin}"][['id', 'text', 'timestamp']].copy()
        
        sentiment_df[f'sent_{coin}']['timestamp'] = pd.to_datetime(sentiment_df[f'sent_{coin}']['timestamp'], unit = 's')

        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].text.apply(func = self._emoji)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._remove_users)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._clean_html)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._remove_links)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._non_ascii)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._lower)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._email_address)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._removeStopWords)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._clean_html)
        sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = self._punct)
        #sentiment_df[f"sent_{coin}"]['new_tweet'] = sentiment_df[f"sent_{coin}"].new_tweet.apply(func = remove_)

    #Sentiment Classification

        tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
        model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest',from_tf=True)

        def sentiment_score(review):
            tokens = tokenizer.encode(review, return_tensors='pt')
            result = model(tokens)
            return int(torch.argmax(result.logits))

        sentiment_df[f"sent_{coin}"]['sentiment'] = sentiment_df[f"sent_{coin}"]['new_tweet'].apply(lambda x: sentiment_score(x[:512]))
        sentiment_df[f"sent_{coin}"].to_csv(os.path.join(self.path,f"{coin}.csv"))
        print(sentiment_df[f"sent_{coin}"].head())
    self.sentiment_df=sentiment_df
    return sentiment_df
  
  def read_csv(self, coin=None):
    df={}
    if coin:
      print(f"You have selected only {coin}, so only {coin} will be returned")
      return {coin: pd.read_csv(os.path.join(self.path,f"{coin}.csv"))}
    
    for coin in self.coins:
      df[f"sent_{coin}"]=pd.read_csv(os.path.join(self.path,f"{coin}.csv"))
    return df
    