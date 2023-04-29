import pandas as pd
import plotly.graph_objects as go

def get_ma_cols(df: pd.DataFrame, strat: dict) -> pd.DataFrame:
    '''
    Add the moving average columns to the dataset, as per the strategy config.
    '''
    
    for ma_type in ['slow', 'fast']:
        
        if strat[ma_type + '_ma_type'] == 'simple':
            df[ma_type] = (
                df[strat[ma_type + '_ma_field']]
                .rolling(strat[ma_type + '_ma_length'])
                .mean()
            )
        elif strat[ma_type + '_ma_type'] == 'exponential':
            df[ma_type] = (
                df[strat[ma_type + '_ma_field']]
                .ewm(span = strat[ma_type + '_ma_length'], adjust = False)
                .mean()
            )
        else:
            raise ValueError(
                'There is no current implementation for the ' + 
                strat[ma_type + '_ma_type'] + ' moving average type.'
            )
    
    return df

# New strategy
STRATEGY = {
    'fast_ma_type': 'exponential',
    'slow_ma_type': 'exponential',
    'fast_ma_field': 'Open',
    'slow_ma_field': 'Close',
    'fast_ma_length': 3,
    'slow_ma_length': 5,
}


TICKER = 'NVDA'
NAME = 'optimal_strat'
LOWER_DATE_FILT = '2017-01-01'

price_data = pd.read_csv(f'data/{TICKER}.csv')

# Calculate the moving average columns with the new strategy
price_data = get_ma_cols(price_data, STRATEGY)
backtest_data = pd.read_csv(f'{TICKER}_{NAME}_backtest.csv')

# Merge price data with backtesting results
price_data['Date'] = pd.to_datetime(price_data['Date'])
backtest_data['bought_on'] = pd.to_datetime(backtest_data['bought_on'])
backtest_data['sold_on'] = pd.to_datetime(backtest_data['sold_on'])

buy_points = price_data[price_data['Date'].isin(backtest_data['bought_on'])]
sell_points = price_data[price_data['Date'].isin(backtest_data['sold_on'])]

# Filter price data and backtesting results for dates after 2022
price_data = price_data[price_data['Date'].dt.year > 2022]
buy_points = buy_points[buy_points['Date'].dt.year > 2022]
sell_points = sell_points[sell_points['Date'].dt.year > 2022]

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=price_data['Date'],
                                     open=price_data['Open'],
                                     high=price_data['High'],
                                     low=price_data['Low'],
                                     close=price_data['Close'])])

# Add fast MA curve
fig.add_trace(go.Scatter(x=price_data['Date'],
                         y=price_data['fast'],
                         mode='lines',
                         name='Fast MA',
                         line=dict(color='blue', width=1)))

# Add slow MA curve
fig.add_trace(go.Scatter(x=price_data['Date'],
                         y=price_data['slow'],
                         mode='lines',
                         name='Slow MA',
                         line=dict(color='orange', width=1)))

# Add buy signals (green markers) for the new strategy
fig.add_trace(go.Scatter(x=buy_points['Date'],
                         y=buy_points['Close'],
                         mode='markers',
                         marker=dict(symbol='circle', size=8, color='green'),
                         name='Buy'))

# Add sell signals (red markers) for the new strategy
fig.add_trace(go.Scatter(x=sell_points['Date'],
                         y=sell_points['Close'],
                         mode='markers',
                         marker=dict(symbol='circle', size=8, color='red'),
                         name='Sell'))

# Customize the chart appearance
fig.update_layout(title=f'{TICKER} - Candlestick Chart with Buy and Sell Signals (New Strategy)',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

# Display the chart
fig.show()
