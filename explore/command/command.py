# %% hra, vra and command
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .select(['pitcher_name', 'pitcher_id', 'game_year', 'pitch_height', 'release_height', 'release_x',
        'arm_angle','p_throws', 'plate_x', 'plate_z', 'hra', 'vra', 'pitch_name'])
    .drop_nulls(subset=['hra', 'vra'])
).collect(engine="streaming")

# pitcher try to hit a few targets with each pitch, can we measure the varbility of when they try to hit
# individual targets. based on the situation can we find an estimate of the location they are trying to hit
# or at least some probablity they are trying to hit a location. can we then find how often they deviate from that 
# location. 

## first lets explore cade horton

# %% find expected vra and hra given arm angle, release height, extension and location

# %% expected location given pitcher, pitch, count, batter handedness, probablity of location

# %% diffrence in expect vra and hra from expected location