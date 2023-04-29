import time
import random
import numba as nb
import numpy as np
import pandas as pd
from copy import deepcopy

# For type hinting
from typing import Tuple

# Global config variables
TRAINING_TICKERS = ['TSM', 'INTC', 'TSM', 'QCOM', 'AVGO', 'TXN']
TESTING_TICKERS = ['NVDA']
LOWER_DATE_FILT = '2017-01-01' # Truncate to consider a fixed length of time
NUM_STRATS = 50 # Number of strategies to try on each evolution
NUM_EVOLVE = 250 # Number of evolutions to perform
KEEP_PERC = 0.3 # Percentage of top models to keep on each evolution

# This is how the fitness is evaluated over a single ticker, e.g. if the
# strategy produces 20 trades, then we may take the mean percentage gained
# over each trade.
# Implemented types: 'mean', 'median', 'compounded'
# Note: 'compounded' means how many multiples of your money would make by 
#       using the strategy on a single ticker.
STRAT_EVAL = 'compounded'

# This is the metric to take from the fitness values of all tickers, e.g. if
# choosing 'min', you select the min fitness over all tickers; this then
# becomes the strategy's fitness.
# Implemented types: 'min', 'mean', 'median'
FITNESS_TYPE = 'mean'

# A minimum of x trades are taken per ticker to prevent overfitting, if you
# seleect 1, then the strategy will find the perfect combination for that 
# ticker (but clearly this is not generalisable to other tickers)
MIN_TRADES = 20

# Strategy parameters to choose from
MA_TYPES = ['simple', 'exponential'] # Types of moving averages to consider
MA_FIELDS = ['Open', 'Low', 'High', 'Close'] # Price fields to choose from
LOWER_MA_LENGTH = 3 # The least length a moving average can have
UPPER_MA_LENGTH = 300 # The maximum length a moving average can 
MAX_PERTURB = 10 # The maximum number to perturb the strategy parameters with

# The strategy to intially perturb, and also to use as a benchmark during
# the testing phase
STARTING_STRAT = {
    'fast_ma_type': 'simple',   
    'slow_ma_type': 'simple',
    'fast_ma_field': 'Close',
    'slow_ma_field': 'Close',
    'fast_ma_length': 10,
    'slow_ma_length': 20,
}


def get_random_strat() -> dict:
    '''
    Generate a fresh random strategy by randomly selecting from the parameters
    definied in the global variables.
    '''
    
    return check_strat({
        'fast_ma_type': random.choice(MA_TYPES),
        'slow_ma_type': random.choice(MA_TYPES),
        'fast_ma_field': random.choice(MA_FIELDS),
        'slow_ma_field': random.choice(MA_FIELDS),
        'fast_ma_length': random.randint(LOWER_MA_LENGTH, UPPER_MA_LENGTH),
        'slow_ma_length': random.randint(LOWER_MA_LENGTH, UPPER_MA_LENGTH),
    })
    

def check_strat(strat: dict) -> dict:
    '''
    This checks if the strategy has valid parameters, and adjusts if not. For
    example, if the slower moving average has a smaller length than the faster
    one, then this is changed to having a larger value.
    '''
    
    for ma_type in ['slow', 'fast']:
        
        if strat[ma_type + '_ma_length'] < LOWER_MA_LENGTH:
            strat[ma_type + '_ma_length'] = LOWER_MA_LENGTH
        elif strat[ma_type + '_ma_length'] > UPPER_MA_LENGTH:
            strat[ma_type + '_ma_length'] = UPPER_MA_LENGTH

    if strat['slow_ma_length'] <= strat['fast_ma_length']:
        strat['slow_ma_length'] = strat['fast_ma_length'] + 1
        
    return strat


def perturb_strat(strat: dict) -> dict:
    '''
    Perturb the parameters of the strategy slightly to generate a new strategy
    '''
    
    for ma_type in ['slow', 'fast']:
        
        strat[ma_type + '_ma_type'] = random.choice(MA_TYPES)
        strat[ma_type + '_ma_field'] = random.choice(MA_FIELDS)
        strat[ma_type + '_ma_length'] += (
            np.random.randint(-MAX_PERTURB, MAX_PERTURB)
        )
    
    return check_strat(strat)


def breed_winning_strats(good_strats: np.array,
                         strats: dict) -> dict:
    '''
    Taking parameters from good/winning strategies and breed a new strategy.
    
    Parameters
    ----------
    good_strats : np.array
        The index values of the best strategies from the evolution
    strats : dict
        The dictionary of all strategies
    '''
    
    new_strat = {}
    
    for param in strats['0'].keys():
        rand_strat_idx = str(random.choice(good_strats))
        new_strat[param] = strats[rand_strat_idx][param]
        
    return check_strat(new_strat)


def init_ga() -> Tuple[list, dict, np.array, np.array]:
    '''
    Initialise the parameters and data needed for the genetic algorithm
    Returns
    -------
    price_data : list
        Price data for all tickers we are optimising over
    strats : dict
        A random set of strategies
    fitness : np_arr
        An array to store the fitness values in for each strategy
    fitness_to_calc : np_arr
        An array to indicate which strategies to calculate the fitness for
    '''
    
    price_data = [
        pd.read_csv(f'data/{ticker}.csv') 
        for ticker in TRAINING_TICKERS
    ]
    
    # Initialise by finding NUM_STRATS strategies which are perturbations from
    # the starting strategy defined in the global variables
    strats = {
        f'{n}': perturb_strat(deepcopy(STARTING_STRAT)) 
        for n in range(0, NUM_STRATS)
    }
    
    # Initialise an empty array to store the fitness values in, col 1 is the
    # idx value of the strategy, and col 2 stores the fitness value
    fitness = np.zeros((NUM_STRATS, 2))
    fitness[:, 0] = np.arange(0, NUM_STRATS)
    
    # Initialise the array to determine which strategies to calculate the
    # fitness for. Initially its all of them, but in the optimisation we only
    # need to calculate for some of them
    fitness_to_calc = np.arange(0, NUM_STRATS)
    
    return price_data, strats, fitness, fitness_to_calc


def get_fitness(price_data: list,
                strats: dict,
                fitness: np.array,
                fitness_to_calc: np.array) -> np.array:
    '''
    Loop over and obtain the fitness for each of the strategies which require
    a new fitness calculation.
    '''
    
    for idx in fitness_to_calc:
        fitness[idx, 1] = strat_fitness(price_data, strats[str(idx)])
    
    return fitness


def strat_fitness(price_data: list,
                  strat: dict, 
                  testing: bool = False) -> float:
    '''
    Calculate the fitness value for a one strategy over all of the price data.
    '''
    
    fitness = []
    for df in price_data:
        
        # Firstly process the price data to include the ma cols (as per the 
        # strategy), and apply the lower date filter.
        df_strat = get_ma_cols(deepcopy(df), strat)
        df_strat = df_strat[df_strat['Date'] > LOWER_DATE_FILT]
        
        # Run the strategy for this ticker's price data, and return a list of
        # percentage gains/losses for each trade.
        trade_res = run_strat(
            df_strat['Open'].values.astype(np.float64),
            df_strat['fast'].values.astype(np.float64),
            df_strat['slow'].values.astype(np.float64),
        )
        
        if STRAT_EVAL == 'mean':
            fitness_val = np.mean(trade_res)
        elif STRAT_EVAL == 'median': 
            fitness_val = np.median(trade_res)
        elif STRAT_EVAL == 'compounded':
            fitness_val = get_compounded(trade_res)
        
        else:
            raise ValueError(
                'The strategy average ' + STRAT_EVAL + 
                ' has not been implemented.'
            )
        
        # This implements the minimum trade per ticker constraint, if we have
        # less than the minimum trades, the fitness value is set to be an
        # extreme low value to strongly encourage against using this strategy
        # NOTE: This is only implemented for training, not for testing
        if trade_res.shape[0] > MIN_TRADES or testing:
            fitness.append(fitness_val)
        else:
            fitness.append(-100)
            
    if FITNESS_TYPE == 'min':
        return np.min(fitness)
    elif FITNESS_TYPE == 'mean':
        return np.mean(fitness)
    elif FITNESS_TYPE == 'median':
        return np.median(fitness)
    else:
        raise ValueError(
            'The fitness type ' + FITNESS_TYPE + 
            ' has not been implemented.'    
        )
    

@nb.jit(nopython = True)
def get_compounded(trade_res: np.array):
    '''
    Get the strategy return as multiples of your initial investment.
    '''
    
    invest = 1
    for perc in trade_res:
        invest = (1+perc)*invest

    return invest


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


@nb.jit(nopython = True)
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
            holding = True
        
        elif holding and ma_sell(day):
            
            trade_res.append(open_prices[day]/bought_at - 1)
            holding = False

    return np.array(trade_res)


def main() -> dict:
    '''
    Main running function for the genetic algorithm
    Returns
    -------
    dict
        The optimised strategy parameters
    '''
    
    # Initialise all the parameters needed to start the evolution
    price_data, strats, fitness, fitness_to_calc = init_ga()
    
    # This defines the number of strategies to change on each evolution
    num_to_change = int((1-KEEP_PERC)*NUM_STRATS)
    
    fitness_save = []
    for evl in range(0, NUM_EVOLVE):
        
        fitness = get_fitness(
            price_data,
            strats,
            fitness,
            fitness_to_calc,
        )
        
        # Rank the strategies, and select the strategies to change
        ranks = fitness[fitness[:, 1].argsort()]
        good_strats = ranks[num_to_change:, 0].astype(np.int32)
        bad_strats = ranks[:num_to_change, 0].astype(np.int32)
        
        # Split the bad strategies into 3 approx equal sets to make changes
        splits = np.array_split(bad_strats, 3)
        
        # Replace some bad strategies with random new ones
        for strat in splits[0]:
            strats[str(strat)] = get_random_strat()
            
        # Add random perturbations to some good strategies
        for strat in splits[1]:
            rand_strat = str(random.choice(good_strats))
            strats[str(strat)] = perturb_strat(deepcopy(strats[rand_strat]))
            
        # Combine good strategies to make new ones
        for strat in splits[2]:
            strats[str(strat)] = breed_winning_strats(
                good_strats,
                deepcopy(strats),
            )
            
        # This shows the optimiser which strats have been changed to calculate
        # the fitness function on the next iteration. This saves us having to
        # recalculate the fitness function for the good strategies and save 
        # computational time
        fitness_to_calc = bad_strats 
        
        # Print out evolution statistics for the best five strategies, this
        # is helpful to see if the optimiser is doing the correct job (i.e.
        # is the fitness being maximised?)
        print(f'\nEvolution {evl}')
        for count, strat in enumerate(np.flipud(good_strats[-5:])):
            print(
                str(count) + '. Strategy: ' +  str(strat) + 
                ', ' + FITNESS_TYPE + ': ' + 
                str(fitness[strat, 1])
            )
        print('----------------------------------------------')
        
        fitness_save.append(deepcopy(fitness))
    
    # Return the most optimal strategy after all evolutions
    return strats[str(good_strats[-1])], fitness_save

if __name__ == '__main__':

    # Run the algorithm to optimise the hyperparameters
    t0 = time.time()
    strat, fitness_save = main()
    print('\nOptimisation time :', str(time.time()-t0))

    # Run the strategy and baseline on the testing tickers
    price_data = [
        pd.read_csv(f'data/{ticker}.csv')
        for ticker in TESTING_TICKERS
    ]
    
    baseline = strat_fitness(price_data, STARTING_STRAT, True)
    optimised = strat_fitness(price_data, strat, True)
    
    print('\n')
    print('Testing values before optimisation:', baseline)
    print('Testing values after optimisation:', optimised)
    print('\nOptimal strategy parameters:', strat)