import numpy as np
import pandas as pd

STRATEGY = {
    'fast_ma_type': 'simple',   
    'slow_ma_type': 'simple',
    'fast_ma_field': 'Close',
    'slow_ma_field': 'Close',
    'fast_ma_length': 10,
    'slow_ma_length': 20,
}

TICKER = 'NVDA'
NAME = 'baseline_strat'
LOWER_DATE_FILT = '2017-01-01'


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


def run_strat(open_prices: np.array,
              fast_ma: np.array,
              slow_ma: np.array) -> np.array:
    '''
    Run the ma crossover strategy. Here, we buy the day after the fast ma 
    crosses from below the slow ma, and sell when the opposite occurs.
    
    Parameters
    ----------
    open_prices : np.array
        The financial instrument open prices on each day
    fast_ma : np.array
        The faster moving average
    slow_ma : np.array
        The slower moving average
        
    Returns
    -------
    trade_res : np.array
        The percentage gained/lost on each trade
    '''
    
    # Flag to determine whether the instrument is currently held or not
    holding = False
    
    # Empty lists to store the results from the strategy
    trade_res = []
    bought_on = []
    sold_on = []
    
    # The logical criteria for if a ma crossover happens, both on the buy and
    # sell side
    ma_buy = lambda day: (
        fast_ma[day-2] < slow_ma[day-2] and
        fast_ma[day-1] > slow_ma[day-1]
    )
    ma_sell = lambda day: (
        fast_ma[day-2] > slow_ma[day-2] and
        fast_ma[day-1] < slow_ma[day-1]
    )
    
    for day in range(2, open_prices.shape[0]):
        
        if not holding and ma_buy(day):
            
            bought_at = open_prices[day]
            bought_on.append(day)
            holding = True
        
        elif holding and ma_sell(day):
            
            trade_res.append(open_prices[day]/bought_at - 1)
            sold_on.append(day)
            holding = False
            
    # We are only interested in stats from completed trades, so if we are still
    # in a trade we delete the last buy
    if holding:
        bought_on = bought_on[:-1]

    return (
        np.array(bought_on),
        np.array(sold_on),
        np.array(trade_res),
    )


if __name__ == '__main__':

    # Read in the price data and calculate the moving average columns
    df = pd.read_csv(f'data/{TICKER}.csv')
    df = get_ma_cols(df, STRATEGY)
    
    # Apply the lower date filter
    df = df[df['Date'] >= LOWER_DATE_FILT]
    
    # Run the strategy
    bought_on, sold_on, trade_res = run_strat(
        df['Open'].values.astype(np.float64),
        df['fast'].values.astype(np.float64),
        df['slow'].values.astype(np.float64),
    )
    
    # Form a dataframe with the trading information
    dates = df['Date'].values
    df_backtest = pd.DataFrame({
        'bought_on': dates[bought_on],
        'sold_on': dates[sold_on],
        'profit': trade_res,
    })
    
    # Save the price data and backtesting results to a csv for plotting
    df_backtest.to_csv(f'{TICKER}_{NAME}_backtest.csv', index = False)