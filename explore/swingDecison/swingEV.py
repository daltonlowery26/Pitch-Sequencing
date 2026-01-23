# %% packages
import polars as pl
import os
import numpy as np
import xgboost as xgb
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.read_parquet()

# %% swing ev
df_s = df.filter(pl.col('swing'))

# %% P(contact | swing traits, pitch embedding)

# %% P(single, double, triple, hr |contact, swing traits, pitch embedding)
