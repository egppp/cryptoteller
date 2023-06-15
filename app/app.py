import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from PIL import Image
from datetime import date, timedelta
import base64


image = Image.open('image.png')
st.image(image)

# Define the available cryptocurrencies
cryptocurrencies = [
    {"name": "BTC-Bitcoin", "current_price": 40000, "predicted_price": 50000, "csv_file": "BTCUSDT.csv"},
    {"name": "ETH-Ethereum", "current_price": 3000, "predicted_price": 3500, "csv_file": "ETHUSDT.csv"},
    {"name": "BNB-Binance", "current_price": 400, "predicted_price": 500, "csv_file": "BNBUSDT.csv"},
    {"name": "XRP-Ripple", "current_price": 0.8, "predicted_price": 1.0, "csv_file": "XRPUSDT.csv"},
    {"name": "ADA-Cardano", "current_price": 2.0, "predicted_price": 2.5, "csv_file": "ADAUSDT.csv"}
]

def plot_miniature_price(crypto):
    # Load the price data for the selected cryptocurrency
    file_path = f"data/{crypto['csv_file']}"
    price_data = pd.read_csv(file_path)

    # Convert 'open_time' column to datetime type
    price_data['open_time'] = pd.to_datetime(price_data['open_time'], unit='ms')

    # Filter the price data for the last year
    ytd_data = price_data.tail(365)

    # Plot the miniature price plot
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(ytd_data['open_time'], ytd_data['open'], color='#275C54')
    ax.set_axis_off()  # Turn off the axis
    ax.margins(0)  # Remove margins
    fig.tight_layout(pad=0)  # Remove padding
    plt.savefig("plot.png", dpi=40)  # Save the plot with low resolution

    # Convert the plot image to a base64 encoded string
    with open("plot.png", "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")

    # Remove the temporary plot image file
    plt.close()
    plt.clf()
    plt.cla()
    plt.gcf().clear()

    return f"<img src='data:image/png;base64,{img_base64}'/>"

def plot_word_cloud(crypto):
    # Get the absolute path of the current working directory
    current_dir = os.getcwd()

    # Define the path of the file relative to the current directory
    file_path = f"data/sentiment/{crypto['name'][0: 3].lower()}.csv"

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
    

    # Display the generated Word Cloud
    fig, ax = plt.subplots()
    ax.imshow(word_cloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)    


# Create a landing page with miniature price plots
def main():
    st.title("Top 5 Cryptocurrencies by Market Capitalisation")

    # Create the table data
    table_data = []
    for crypto in cryptocurrencies:
        row = [
            crypto['name'],
            f"${crypto['current_price']}",
            f"${crypto['predicted_price']}",
            plot_miniature_price(crypto),
            f"<a href='#{crypto['name'].replace(' ', '-')}'>Crypto Page</a>"
        ]
        table_data.append(row)

    # Create a custom HTML table with embedded images
    table_html = "<table><tbody>"
    for row in table_data:
        table_html += "<tr>"
        for item in row:
            if item.startswith("<img src="):  # Embed the image in the cell
                table_html += f"<td>{item}</td>"
            else:
                table_html += f"<td>{item}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"

    # Display the HTML table
    st.markdown(table_html, unsafe_allow_html=True)


# Create a cryptocurrency-specific page
def crypto_page(crypto):
    st.title(f"{crypto['name']} Price Analysis")

    # Display the current price and predicted price
    st.write(f"Current Price: ${crypto['current_price']}")
    st.write(f"Predicted Price: ${crypto['predicted_price']}")

    # Display the miniature price graph
    plot_miniature_price(crypto)

    # Display the word cloud
    plot_word_cloud(crypto)
    

if __name__ == "__main__":
    main()


