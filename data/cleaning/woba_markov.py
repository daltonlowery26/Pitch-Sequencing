# woba value derivied by applying markov chains
# %% package
import polars as pl
import numpy as np
import os
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/')

# %% load data
pitch = (pl.scan_csv('cleaned_data/pitch_2015_2026.csv')
    # add count 
    .filter((pl.col('balls') != 4) & (pl.col('strikes') != 3))
    .with_columns(count = pl.concat_str([pl.col('balls'), pl.col('strikes')],separator="-").alias('count'))
)

# %% count dict
next_counts = {
    '0-0': ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2'],
    '1-0': ['1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2'],
    '0-1': ['0-1', '1-1', '0-2', '1-2', '2-1', '2-2', '3-1', '3-2'],
    '1-1': ['1-1', '1-2', '2-1', '2-2', '3-1', '3-2'],
    '0-2': ['0-2', '1-2', '2-2', '3-2'],
    '1-2': ['1-2', '2-2', '3-2'],
    '2-0': ['2-0', '2-1', '2-2', '3-0', '3-1', '3-2'],
    '2-1': ['2-1', '2-2', '3-1', '3-2'],
    '3-0': ['3-0', '3-1', '3-2'],
    '2-2': ['2-2', '3-2'],
    '3-1': ['3-1', '3-2'],
    '3-2': ['3-2']
}

# mean woba value of each count
mean_woba_by_count = pitch.group_by(['count', 'game_year']).agg([
    pl.col('woba_value').mean().alias('woba_value'),
    pl.col('count').len().alias('counts')
]).collect(engine="streaming")
mean_woba_by_count.head()

# %% state prob
# start states
states = list(next_counts.keys())
# sort by count
pitch = pitch.sort(by=['game_pk', 'at_bat_number', 'pitch_number'])
# what is the next count
transitions = pitch.with_columns(
    next_count = pl.col('count')
    .shift(-1)
    .over(['game_pk', 'at_bat_number'])
    .fill_null('end')
)
# amount of prev to next
transition_counts = (
    transitions
    .group_by(['count', 'next_count', 'game_year'])
    .len()
    .rename({'len': 'occ'})
)
# prob of count going to next state
final_probs = (
    transition_counts.with_columns(
        start_count = pl.col('occ').sum().over(['count', 'game_year'])
    )
    .with_columns(
        prob = pl.col('occ') / pl.col('start_count')
    )
    .sort(['count', 'next_count'])
).collect(engine="streaming")

final_probs.head()

# %% wOBA value of state
def calc_woba_values(transition, terminal_values):
    # transtion has form of start state, next state, prob
    # terminal value has start, next, value
    
    # starts and next states
    all_starts = transition.select(pl.col("count").unique()).to_series().to_list()
    
    # terminal states, len of start states
    terminal = ["end"]
    transient = sorted(list(all_starts))
    n = len(all_starts)
    
    # map names to index
    t_idx = {state: i for i, state in enumerate(transient)}
    
    # q matrix and b vector
    q = np.zeros((n, n))
    b = np.zeros(n)
    
    for row in transition.iter_rows(named=True):
        # extract state, next state, and prob
        s = row['count'] 
        ns = row['next_count']
        p = row['prob']
        # name of state
        i = t_idx[s]
        # if the next state is terminal
        if ns in terminal:
            # get the value, mult it by prob
            val = terminal_values.get(s)
            b[i] += p*val
        else:
            # if not add to transiton matrix
            if ns in t_idx:
                j = t_idx[ns]
                q[i, j] = p
    i = np.eye(n) # identity matrix
    v_trans = np.linalg.solve(i - q, b) # solve linear system
    
    return {state:val for state, val in zip(transient, v_trans)}
    
# format for solving
years = final_probs['game_year'].unique()
yearly_wobas = []
for year in years:
    state_to_next = final_probs.filter(pl.col('game_year') == year).select(pl.col('count'), pl.col('next_count'), pl.col('prob')) 
    woba_year = mean_woba_by_count.filter(pl.col('game_year') == year)
    terminal_values = dict(zip(
        woba_year["count"], 
        woba_year["woba_value"]
    ))
    
    woba_value = calc_woba_values(state_to_next, terminal_values)
    values = {
        'year':year,
        'values':woba_value
    }
    yearly_wobas.append(values)

# %% export woba
yearly = pl.from_dicts(yearly_wobas)
yearly = yearly.unnest('values')
yearly.write_csv("raw_data/woba.csv")




        
    
    
    
    
        
        
    
        
