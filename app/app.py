import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from PIL import Image

image = Image.open('image.png')
st.image(image)

# Get the user-selected cryptocurrency
cryptocurrencies = st.selectbox("Which crypto?", options=["BTC-Bitcoin", "ETH-Ethereum", "BNB-Binance", "XRP-Ripple", "ADA-Cardano"])

# Map the selected cryptocurrency to the corresponding CSV file name
crypto_mapping = {
    "BTC-Bitcoin": "BTCUSDT.csv",
    "ETH-Ethereum": "ETHUSDT.csv",
    "BNB-Binance": "BNBUSDT.csv",
    "XRP-Ripple": "XRPUSDT.csv",
    "ADA-Cardano": "ADAUSDT.csv"
}

# Get the file path for the selected cryptocurrency
file_path = f"data/{crypto_mapping[cryptocurrencies]}"

# Load the price data for the selected cryptocurrency
price_data = pd.read_csv(file_path)

# Convert 'Date' column to datetime type
price_data['open_time'] = pd.to_datetime(price_data['open_time'], unit='ms')

# Define the available date range
min_date = price_data['open_time'].min().date()
max_date = price_data['open_time'].max().date()

# Define the default value for the "From date" input
default_from_date = min_date

# Get the date range from the user
from_date = st.date_input('From date', value=default_from_date, min_value=min_date, max_value=max_date)
to_date = st.date_input('To date', min_value=from_date, max_value=max_date)

# Plot the price graph
fig, ax = plt.subplots()
ax.plot(filtered_data['open_time'], filtered_data['open'])
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Price Graph')
st.pyplot(fig)

# Get the absolute path of the current working directory
current_dir = os.getcwd()

# Define the path of the file relative to the current directory
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]

for s in symbols:
    coin = (s[0: 3].lower())
    file_path = f"data/sentiment/{coin}.csv"

    # Join the current directory path with the file path to get the absolute file path
    absolute_path = os.path.join(current_dir, file_path)

    # Get the relative path by using os.path.relpath()
    relative_path = os.path.relpath(absolute_path, start=current_dir)

    df = pd.read_csv(relative_path)

    text = " ".join(str(tweet) for tweet in df.new_tweet)

    # Define the list of keywords to exclude
    excluded_keywords = ["bitcoin", "btc", "ethereum", "eth", "binance", "bnb", "ripple", "xrp", "cardano", "ada", "crypto", "cryptocurrency", "cryptocurrencies"]

    # Remove excluded keywords from the text
    text = ' '.join(str(word) for word in text.split() if word not in excluded_keywords)

    # Create the word cloud
    #, font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
    word_cloud = WordCloud(collocations = False, background_color = 'white', font_path='/System/Library/Fonts/Supplemental/Arial.ttf', colormap='BrBG', width=800, height=400).generate(text)
    
    def main():
        # Display the generated Word Cloud
        fig, ax = plt.subplots()
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    if __name__ == "__main__":
        main()