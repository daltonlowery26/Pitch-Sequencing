# %% packages
import polars as pl
import numpy as np
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

input = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .select(['pitcher_name', 'pitch_name', 'game_year', 'pitcher_id',
            'hra', 'vra', 'effective_speed', 'release_speed', 'arm_angle', 
            'release_height', 'release_extension', 'release_x', 'vx0', 'ax', 'vz0', 'az', 'pitch_value'])).collect(engine="streaming")

# %% feat
# tth 
input = input.with_columns(
    tth = ((60.5 - pl.col('release_extension')) / (pl.col('release_speed') * 1.46667))
)
# decison point
input = input.with_columns(
    dp = pl.col('tth') - .150 # batter has to decide to swing 150ms before
)
# delta x, displacement: vx0t + 1/2axt^2
input = input.with_columns(
    deltax = pl.col('vx0') * pl.col('dp') + (1/2) * pl.col('ax') * pl.col('dp')**2
)
# delta z, displacment but for z
input = input.with_columns(
    deltaz = pl.col('vz0') * pl.col('dp') + (1/2) * pl.col('az') * pl.col('dp')**2
)
input.head()

# %% export
input.write_parquet('cleaned_data/embed/pitch.parquet')
