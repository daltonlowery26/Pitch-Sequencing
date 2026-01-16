# evaluating the quality of a hitters swing decisons. what priors is a hitter opperating based on? 
# priors for a sd
# -pitcher quality; - their own ability; -game state?

# %% packages
import polars as pl
import os
import numpy as np
import xgboost as xgb
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .filter(pl.col('swing')
    )
)