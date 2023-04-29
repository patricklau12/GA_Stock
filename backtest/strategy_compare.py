import plotly.graph_objects as go
import pandas as pd

TICKER = 'NVDA'
NAME1 = 'baseline_strat'
NAME2 = 'optimal_strat'
LOWER_DATE_FILT = '2017-01-01'

BASELINE_STRATEGY = {
    'fast_ma_type': 'simple',   
    'slow_ma_type': 'simple',
    'fast_ma_field': 'Close',
    'slow_ma_field': 'Close',
    'fast_ma_length': 10,
    'slow_ma_length': 20,
}

OPTIMAL_STRATEGY = {
    'fast_ma_type': 'exponential',
    'slow_ma_type': 'exponential',
    'fast_ma_field': 'Open',
    'slow_ma_field': 'Close',
    'fast_ma_length': 3,
    'slow_ma_length': 5,
}

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

# Read the price data and backtesting results
price_data = pd.read_csv(f'data/{TICKER}.csv')
backtest_data = pd.read_csv(f'{TICKER}_{NAME1}_backtest.csv')
new_backtest_data = pd.read_csv(f'{TICKER}_{NAME2}_backtest.csv')

# Calculate the moving average columns
price_data = get_ma_cols(price_data, BASELINE_STRATEGY)
price_data2 = get_ma_cols(price_data, OPTIMAL_STRATEGY)

# Merge price data with backtesting results
price_data['Date'] = pd.to_datetime(price_data['Date'])
backtest_data['bought_on'] = pd.to_datetime(backtest_data['bought_on'])
backtest_data['sold_on'] = pd.to_datetime(backtest_data['sold_on'])
new_backtest_data['bought_on'] = pd.to_datetime(new_backtest_data['bought_on'])
new_backtest_data['sold_on'] = pd.to_datetime(new_backtest_data['sold_on'])

# Filter price_data to include data starting from 2017
price_data = price_data[price_data['Date'] >= '2017-01-01']

# Define the start_date
start_date = price_data['Date'].min()

# Calculate the buy and hold cumulative return for each point in time
price_data['buy_and_hold_return'] = price_data['Close'] / price_data.loc[price_data['Date'] == start_date, 'Close'].values[0] * 100

# Calculate the cumulative equity curve for the moving average crossover strategy
backtest_data['crossover_equity'] = (backtest_data['profit'] + 1).cumprod() * 100
new_backtest_data['new_crossover_equity'] = (new_backtest_data['profit'] + 1).cumprod() * 100

# Merge the equity curve of the moving average crossover strategy to price_data
price_data = price_data.merge(backtest_data[['sold_on', 'crossover_equity']], left_on='Date', right_on='sold_on', how='left')
price_data['crossover_equity'] = price_data['crossover_equity'].fillna(method='ffill')

# Create the equity curve plot
fig = go.Figure()

# Add the equity curve for the original moving average crossover strategy
fig.add_trace(go.Scatter(x=price_data['Date'],
                         y=price_data['crossover_equity'],
                         mode='lines',
                         name='Original Moving Average Crossover Equity',
                         line=dict(color='blue', width=2)))

# Add the equity curve for the new moving average crossover strategy
fig.add_trace(go.Scatter(x=new_backtest_data['sold_on'],
                         y=new_backtest_data['new_crossover_equity'],
                         mode='lines',
                         name='New Moving Average Crossover Equity',
                         line=dict(color='purple', width=2)))

# Add the equity curve for the buy and hold strategy
fig.add_trace(go.Scatter(x=price_data['Date'],
                         y=price_data['buy_and_hold_return'],
                         mode='lines',
                         name='Buy and Hold Equity',
                         line=dict(color='orange', width=2)))

# Customize the chart appearance
fig.update_layout(title='Equity Curve Comparison: Two Moving Average Crossover Strategies vs. Buy and Hold',
                  xaxis_title='Date',
                  yaxis_title='Equity',
                  xaxis_rangeslider_visible=False)

# Display the chart
fig.show()
