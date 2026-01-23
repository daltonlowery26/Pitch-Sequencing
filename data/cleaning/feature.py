# Adding features, selecting features
# %% packages
import os
import numpy as np
import polars as pl
os.chdir(
    "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/"
)

# %% select pitches
pitches = [
    "4-Seam Fastball","Curveball","Sinker","Forkball","Slurve","Knuckle Curve",
    "Slider","Cutter","Changeup","Sweeper","Knuckleball", "Split-Finger",
]
pitch = (
    pl.scan_csv("cleaned_data/pitch_2015_2026.csv")
    .filter(pl.col("pitch_name").is_in(pitches))
)

# vaa
vy_f = -(pl.col("vy0").pow(2) - (2 * pl.col("ay") * (50 - (17 / 12)))).sqrt()
t = (vy_f - pl.col("vy0")) / pl.col("ay")
vz_f = pl.col("vz0") + (pl.col("az") * t)
pitch = pitch.with_columns(vaa=-(vz_f / vy_f).arctan() * (180 / np.pi))

# haa
vx_f = pl.col("vx0") + pl.col("az") * t
pitch = pitch.with_columns(haa=-(vx_f / vy_f).arctan() * (180 / np.pi))

# based on m_rosen kirby index release angles
def calculate_VRA(vy0, ay, release_extension, vz0, az):
    d_release = 60.5 - release_extension - 50
    vy_s = -((vy0**2 + 2 * ay * d_release).sqrt())
    t_s = (vy0 - vy_s) / ay
    vz_s = vz0 - az * t_s
    VRA = - (vz_s / vy_s).arctan() * (180 / np.pi)
    return VRA
    
def calculate_HRA(vy0, ay, release_extension, vx0, ax):
    d_release = 60.5 - release_extension - 50
    vy_s = -((vy0**2 + 2 * ay * d_release).sqrt())
    t_s = (vy0 - vy_s) / ay
    vx_s = vx0 - ax * t_s
    HRA = - (vx_s / vy_s).arctan() * (180 / np.pi)
    return HRA

# hra and vra for each pitch
pitch = pitch.with_columns(pl.col('release_extension').fill_null(pl.col('release_extension').mean()))
pitch = pitch.with_columns(
    calculate_VRA(pl.col('vy0'), pl.col('ay'), pl.col('release_extension'), pl.col('vz0'), pl.col('az')).alias('vra'),
    calculate_HRA(pl.col('vy0'), pl.col('ay'), pl.col('release_extension'), pl.col('vx0'), pl.col('ax')).alias('hra')
)

# ivb
pitch = pitch.rename({"pfx_z": "ivb"})
# ihb
pitch = pitch.rename({"pfx_x": "ihb"})
# arm angle nas
pitch = pitch.with_columns(
    pl.col('arm_angle').cast(pl.Float32)
)
pitch = pitch.collect(engine="streaming")
pitch = pitch.with_columns(
    pl.col('arm_angle').fill_null(
        pl.col('arm_angle').mean().over(['pitcher_id', 'pitch_name', 'game_year']) + 
        ( 
        pl.Series(np.random.normal(size=len(pitch))) * pl.col('arm_angle').std().over(['pitcher_id', 'pitch_name', 'game_year'])
        )
    )
)

# %% physical traits csv
pitch.write_csv('cleaned_data/pitch_ft_1526.parquet')

