# %% packages
import polars as pl
import numpy as np
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.scan_csv('cleaned_data/pitch_2015_2026.csv').select(['pitcher_name', 'pitcher_id', 'game_year', 'batter_name', 'delta_run_exp', 'pitch_value'])
metrics = pl.read_csv('raw_data/general/pitcher_preformance.csv')
print(metrics.columns)

# %% merge with preformance stats
df = df.drop_nulls(subset=['pitcher_id', 'game_year', 'pitcher_name', 'delta_run_exp', 'pitch_value'])
df = df.group_by(['pitcher_id', 'game_year', 'pitcher_name']).agg(
    run_value = pl.col('delta_run_exp').sum(),
    woba = pl.col('pitch_value').sum(),
    pitches = pl.col('pitch_value').count()
)
df = df.collect(engine="streaming")
df = df.join(metrics, right_on=['MLBAMID', 'Season'], left_on=['pitcher_id', 'game_year'])
df.head()

# %% corrs
print(df.select(pl.corr(pl.col('SIERA'), pl.col('woba'))))
print(df.select(pl.corr(pl.col('FIP-'), pl.col('woba'))))
print(df.select(pl.corr(pl.col('SIERA'), pl.col('run_value'))))
print(df.select(pl.corr(pl.col('FIP-'), pl.col('run_value'))))
