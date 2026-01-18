# evaluating the quality of a hitters swing decisons. what priors is a hitter opperating based on? 
# priors for a sd
# given a pitch what is the most likely decison a hitter will make?
# punish less for common decisons, hard pitches
# considering the decison a hitter make what is gain in expected value above average

# architecture
# given the pitch embedding, swing traits, how do we 
# 


# %% packages
import polars as pl
import os
import numpy as np
import xgboost as xgb
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .select(['pitcher_name','pitcher_id', 'batter_name', 'batter_id', 'game_year', 'pitch_name',
            'count', 'abs_strike', 'swing', 'hra', 'vra', 'release_height', 'release_x','plate_x', 'plate_z' 'count_value', 'pitch_value', 
            'attack_angle', 'attack_direction', 'bat_speed', 'swing_path_tilt'])
)