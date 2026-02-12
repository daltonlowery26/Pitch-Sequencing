# %% packages
import polars as pl
import numpy as np
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
input = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
        .select(pl.col(['pitcher_name', 'pitcher_id', 'pitch_name','p_throws', 'b_stand', 
                        'at_bat_number', 'pitch_number', 'n_thruorder_pitcher', 'count', 'swing', 'zone']))
).collect(engine="streaming")

# %%