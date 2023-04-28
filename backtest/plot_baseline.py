import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
STRATEGY = {
    'fast_ma_type': 'simple',   
    'slow_ma_type': 'simple',
    'fast_ma_field': 'Close',
    'slow_ma_field': 'Close',
    'fast_ma_length': 10,
    'slow_ma_length': 20,
}

# Assuming 'data' is a DataFrame with 'Date', 'Open', 'High', 'Low', 'Close' columns
data = pd.read_csv('./data/NVDA.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data from 2022 to the current date
start_date = pd.Timestamp(year=2022, month=1, day=1)
end_date = pd.Timestamp.now()
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Set 'Date' as index
data.set_index('Date', inplace=True)

# Calculate the moving averages
data['fast_ma'] = data[STRATEGY['fast_ma_field']].rolling(window=STRATEGY['fast_ma_length']).mean()
data['slow_ma'] = data[STRATEGY['slow_ma_field']].rolling(window=STRATEGY['slow_ma_length']).mean()

# Identify trade periods (when the fast moving average crosses above the slow moving average)
data['trade_signal'] = np.where(data['fast_ma'] > data['slow_ma'], 1, 0)
data['trade_period'] = data['trade_signal'].diff().eq(1)

# Create the moving average plot
ap_mav = [mpf.make_addplot(data['fast_ma'], color='blue', linestyle='--', panel=0),
          mpf.make_addplot(data['slow_ma'], color='red', linestyle='--', panel=0)]

# Plot the candlestick chart with moving averages and trade periods
fig, axes = mpf.plot(data,
                     type='candle',  # Candlestick chart
                     style='yahoo',  # Yahoo Finance style
                     title='Stock Price with Moving Average Strategy (2022 - Present)',
                     ylabel='Price',
                     addplot=ap_mav,
                     show_nontrading=True,
                     returnfig=True)

# Highlight the background of trading periods
ax = axes[0]
trade_periods = data.index[data['trade_signal'] == 1]
for trade_period in trade_periods:
    ax.fill_between(data.index, data['fast_ma'], data['slow_ma'], where=(data.index >= trade_period), facecolor='green', alpha=0.2, interpolate=True)

# Show the plot
plt.show()