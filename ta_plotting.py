from technical_analysis import ta_symbols 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

#Stochastic Oscillator

# Create our primary chart
# the rows/cols arguments tell plotly we want two figures
fig = make_subplots(rows=2, cols=1)  
# Create our Candlestick chart with an overlaid price line
fig.append_trace(
    go.Candlestick(
        x=ta_symbols['ta_btc']['open_time'],
        open=ta_symbols['ta_btc']['open'],
        high=ta_symbols['ta_btc']['high'],
        low=ta_symbols['ta_btc']['low'],
        close=ta_symbols['ta_btc']['close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False
    ), row=1, col=1  # <------------ upper chart
)
# price Line
fig.append_trace(
    go.Scatter(
        x=ta_symbols['ta_btc']['open_time'],
        y=ta_symbols['ta_btc']['open'],
        line=dict(color='#ff9900', width=1),
        name='open',
    ), row=1, col=1  # <------------ upper chart
)
# Fast Signal (%k)
fig.append_trace(
    go.Scatter(
        x=ta_symbols['ta_btc']['open_time'],
        y=ta_symbols['ta_btc']['%K'],
        line=dict(color='#ff9900', width=2),
        name='fast',
    ), row=2, col=1  #  <------------ lower chart
)
# Slow signal (%d)
fig.append_trace(
    go.Scatter(
        x=ta_symbols['ta_btc']['open_time'],
        y=ta_symbols['ta_btc']['%D'],
        line=dict(color='#000000', width=2),
        name='slow',
    ), row=2, col=1  #<------------ lower chart
)
# Extend our y-axis a bit
fig.update_yaxes(range=[-10, 110], row=2, col=1)
# Add upper/lower bounds
fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
# Add overbought/oversold
fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)
fig.update_layout(layout)
# View our chart in the system default HTML viewer (Chrome, Firefox, etc.)
fig.show()

#MACD - Moving Average Convergence Divergence
fig = make_subplots(rows=2, cols=1)
# price Line
fig.append_trace(
    go.Scatter(
        x=ta_symbols['ta_btc']['open_time'],
        y=ta_symbols['ta_btc']['open'],
        line=dict(color='#ff9900', width=1),
        name='open',
        # showlegend=False,
        legendgroup='1',
    ), row=1, col=1
)
# Candlestick chart for pricing
fig.append_trace(
    go.Candlestick(
        x=ta_symbols['ta_btc']['open_time'],
        open=ta_symbols['ta_btc']['open'],
        high=ta_symbols['ta_btc']['high'],
        low=ta_symbols['ta_btc']['low'],
        close=ta_symbols['ta_btc']['close'],
        increasing_line_color='#ff9900',
        decreasing_line_color='black',
        showlegend=False
    ), row=1, col=1
)
# Fast Signal (%k)
fig.append_trace(
    go.Scatter(
        x=ta_symbols['ta_btc']['open_time'],
        y=ta_symbols['ta_btc']['MACD_12_26_9'],
        line=dict(color='#ff9900', width=2),
        name='macd',
        # showlegend=False,
        legendgroup='2',
    ), row=2, col=1
)
# Slow signal (%d)
fig.append_trace(
    go.Scatter(
        x=ta_symbols['ta_btc']['open_time'],
        y=ta_symbols['ta_btc']['MACDs_12_26_9'],
        line=dict(color='#000000', width=2),
        # showlegend=False,
        legendgroup='2',
        name='signal'
    ), row=2, col=1
)
# Colorize the histogram values
colors = np.where(ta_symbols['ta_btc']['MACDh_12_26_9'] < 0, '#000', '#ff9900')
# Plot the histogram
fig.append_trace(
    go.Bar(
        x=ta_symbols['ta_btc']['open_time'],
        y=ta_symbols['ta_btc']['MACDh_12_26_9'],
        name='histogram',
        marker_color=colors,
    ), row=2, col=1
)
# Make it pretty
layout = go.Layout(
    plot_bgcolor='#efefef',
    # Font Families
    font_family='Monospace',
    font_color='#000000',
    font_size=20,
    xaxis=dict(
        rangeslider=dict(
            visible=False
        )
    )
)
# Update options and show plot
fig.update_layout(layout)
fig.show()