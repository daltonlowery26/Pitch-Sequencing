# %% packages
import polars as pl
import numpy as np
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

input = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .select(['pitcher_name', 'pitcher_id', 'batter_name', 'batter_id', 'pitch_number', 'pitch_name','game_pk', 'at_bat_number', 
                    'hra', 'vra', 'effective_speed', 'arm_angle', 'release_extension', 'release_speed',
                    'vx0', 'ax', 'vz0', 'az', 'release_height', 'release_x', 'ay', 'description', 'plate_x', 'plate_z',
                    'delta_run_exp', 'pitch_value', 'swing', 'abs_strike', 'count', 'attack_angle', 'attack_direction', 
                    'swing_path_tilt', 'bat_speed', 'swing_length']).collect(engine="streaming")
)
# %% feat
# tth 
input = input.with_columns(
    tth = ((60.5 - pl.col('release_extension')) / (pl.col('release_speed') * 1.46667))
)
# decison point
input = input.with_columns(
    dp = pl.col('tth') - .150 # batter has to decide to swing 150ms before
)
input = input.with_columns(
    mid = pl.col('tth') - .300
)
# delta x, displacement: vx0t + 1/2axt^2
input = input.with_columns(
    deltax = pl.col('vx0') * pl.col('dp') + (1/2) * pl.col('ax') * pl.col('dp')**2
)
# delta z, displacment but for z
input = input.with_columns(
    deltaz = pl.col('vz0') * pl.col('dp') + (1/2) * pl.col('az') * pl.col('dp')**2
)
# midflight travel
input = input.with_columns(
    midx = pl.col('vx0') * pl.col('mid') + (1/2) * pl.col('ax') * pl.col('mid')**2
)
# delta z, displacment but for z
input = input.with_columns(
    midz = pl.col('vz0') * pl.col('mid') + (1/2) * pl.col('az') * pl.col('mid')**2
)

# %% export
input.write_parquet('cleaned_data/embed/pitch.parquet')
