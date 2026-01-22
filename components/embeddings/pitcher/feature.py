# remove the effect of pitch location from all measures

# %% packages
import polars as pl
import numpy as np
import os
from sklearn.linear_model import LinearRegression
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
        .select(['pitcher_name', 'pitcher_id', 'p_throws', 'pitch_name', 'game_year',
        'vaa', 'haa', 'effective_speed', 'release_speed', 'az', 'ay', 'ax', 'release_extension', 
        'arm_angle', 'release_height', 'release_x', 'plate_x', 'plate_z'])
        .drop_nulls()).collect(engine="streaming")


print(df.select(pl.corr("plate_z", "arm_angle")).item())
print(df.select(pl.corr("plate_x", "ax")).item())

# %% az adj for gravity
df = df.with_columns(
    pl.col('az') + 32.174
)

# %% vaa adusted
X = df.select(['plate_z'])
y = df.select(['vaa'])
vaa_m = LinearRegression()
vaa_m.fit(X, y)
expected_vaa = vaa_m.predict(X).flatten()
df = df.with_columns(
    (pl.col('vaa') - expected_vaa).alias('vaa_diff')
)

# %% haa adusted
X = df.select(['plate_x'])
y = df.select(['haa'])
vaa_m = LinearRegression()
vaa_m.fit(X, y)
expected_vaa = vaa_m.predict(X).flatten()
df = df.with_columns(
    (pl.col('haa') - expected_vaa).alias('haa_diff')
)

# %% 
df = df.drop_nulls()
# %% adj export
df.write_parquet('cleaned_data/embed/loc_adj.parquet')
