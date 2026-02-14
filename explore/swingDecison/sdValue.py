# %% packages
from eventProb.foulProb import foulPredict
from eventProb.takeProb import takePredict
from eventProb.whiffProb import contactPredict
from bipValue import bipValue
import os
import polars as pl
os.chdir('/Users/daltonlowery/Desktop/projects/Optimal Pitch/data')

# %% load data
df = pl.read_parquet('cleaned_data/metrics/xswing/xtraitContact.parquet')
swing_traits = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
df = df.with_columns(traits = pl.concat_list(swing_traits))
df = df.with_columns(inter = pl.concat_list('intercept_x', 'intercept_y'))

# %% forward passes
# sd value = takeValue - swingValue
# takeValue = diff in EV from count state to next
# swingValue = prob of contact * (prob of foul * foulValue) + (prob of bip * bipValue) 
