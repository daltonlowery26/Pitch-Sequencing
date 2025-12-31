# pitcher try to hit a few targets with each pitch, can we measure the varbility of when they try to hit
# individual targets. based on the situation can we find an estimate of the location they are trying to hit
# or at least some probablity they are trying to hit a location. can we then find how often they deviate from that 
# location. 
# %% hra, vra and command
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .select(['pitcher_name', 'pitcher_id', 'game_year', 'b_stand', 'count', 'release_height', 'release_x',
        'arm_angle','p_throws', 'plate_x', 'plate_z', 'hra', 'vra', 'pitch_name'])
    .drop_nulls(subset=['hra', 'vra'])
).collect(engine="streaming")
# is a pitcher ahead or behind in the count 
ahead_in_count = ['0-1', '0-2', '1-2', '2-2']
df = df.with_columns(
    ahead = pl.col('count').is_in(ahead_in_count)
)

# %% plotting pitch loc and intention
def plot_pitch_intent(pitch_data, targets):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # plot pitches
    ax.scatter(pitch_data[:, 0], pitch_data[:, 1], 
               alpha=0.3, color='steelbablue', s=15, label='Actual Pitches')
    # plot intended target
    ax.scatter(targets[:, 0], targets[:, 1], 
               color='red', marker='X', s=200, edgecolor='white', linewidth=1.5, 
               zorder=10, label='Intended Target')
    # zone
    sz_width = 17 / 12
    sz_left = -sz_width / 2
    sz_bottom = 1.5
    sz_height = 2.0
    
    zone = Rectangle((sz_left, sz_bottom), sz_width, sz_height,
                     linewidth=2, edgecolor='black', facecolor='none', 
                     linestyle='--', label='Strike Zone')
    ax.add_patch(zone)

    # formatting
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 5.0)
    ax.set_xlabel('Horizontal Location (ft)')
    ax.set_ylabel('Vertical Location (ft)')
    ax.set_title('Inferred Intended Pitch Locations')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# %% vra, hra miss
def vra_hra_miss(pitchers, pitches, b_stand, df=df):
    # predicted vra and hra given location and release x and release height
    lin_R = (df
        .filter(pl.col('p_throws') == 'R')
        .select(['plate_x', 'plate_z', 'release_x', 'release_height', 'vra', 'hra'])
        .drop_nulls()
    )
    
    X_train = lin_R.select(['plate_x', 'plate_z', 'release_x', 'release_height']).to_numpy()
    vra_train = lin_R.select(['vra']).to_numpy()
    hra_train = lin_R.select(['hra']).to_numpy()
    # trained models
    vra_model_R = LinearRegression().fit(X_train, vra_train)
    hra_model_R = LinearRegression().fit(X_train, hra_train)

    # batch filter all data
    cohort = df.filter(
        (pl.col('pitcher_name').is_in(pitchers)) & 
        (pl.col('pitch_name').is_in(pitches)) &
        (pl.col('game_year') == 2025) &
        (pl.col('b_stand') == b_stand) &
        (pl.col('ahead'))
    )

    misses_accumulator = []

    # loop through pitchers and pitches
    for keys, indiv_pitch in cohort.group_by(['pitcher_name', 'pitch_name']):
        current_pitcher, current_pitch = keys
        
        # skip if less than 50, only 5 misses
        if len(indiv_pitch) < 50: 
            continue
        # all pitch locations
        indiv_loc = indiv_pitch.select(['plate_x', 'plate_z']).to_numpy()
        # fit gmm
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=26)
        gmm.fit(indiv_loc)
        intended_locations = gmm.means_
        
        # misses
        scores = gmm.score_samples(indiv_loc)
        threshold = np.percentile(scores, 100) # take 20% biggest misses
        
        # filter only for outliers
        is_outlier = scores < threshold
        
        if not np.any(is_outlier):
            raise Exception()
        # only keep large misses
        pitch_misses = indiv_pitch.filter(pl.lit(is_outlier))

        # find closest target
        miss_coords = pitch_misses.select(['plate_x', 'plate_z']).to_numpy()
        diffs = miss_coords[:, np.newaxis, :] - intended_locations
        dists = np.linalg.norm(diffs, axis=2)
        closest_indices = np.argmin(dists, axis=1)
        closest_means = intended_locations[closest_indices]
        
        # assumed target for misses
        pitch_misses = pitch_misses.with_columns(
            pl.Series("target_x", closest_means[:, 0]),
            pl.Series("target_z", closest_means[:, 1])
        )
        
        misses_accumulator.append(pitch_misses)

    # combine all missed pitches
    all_misses = pl.concat(misses_accumulator)
    
    # preds based on trageted location
    X_pred = all_misses.select(['target_x', 'target_z', 'release_x', 'release_height']).to_numpy()
    
    # predict
    target_vras = vra_model_R.predict(X_pred).flatten()
    target_hras = hra_model_R.predict(X_pred).flatten()
    
    # diffs and output
    final_results = (all_misses
        .with_columns(
            pl.Series("target_vra", target_vras),
            pl.Series("target_hra", target_hras)
        )
        .with_columns(
            (pl.col('vra') - pl.col('target_vra')).abs().alias('vra_miss'),
            (pl.col('hra') - pl.col('target_hra')).abs().alias('hra_miss')
        )
        .group_by(['pitcher_name', 'pitch_name'])
        .agg(
            pl.col('vra_miss').mean().alias('avg_vra_miss'),
            pl.col('hra_miss').mean().alias('avg_hra_miss'),
            pl.len().alias('miss_count')
        )
        .sort('pitcher_name')
    )

    return final_results

# %% location x miss

# %% command grades 
pitchers = (df
    .filter(pl.col('p_throws') == 'R')
    .filter(pl.col('game_year') > 2024)
    .filter(pl.len().over('pitcher_id') > 1000)
)

pitcher_name = pitchers['pitcher_name'].unique().to_list()
pitch_type =  pitchers['pitch_name'].unique().to_list()
# simple vra and hra miss
command_grades = vra_hra_miss(pitchers=pitcher_name, pitches=pitch_type, b_stand='R')

# adj grades based on averages
final_grades = (command_grades
    # avg miss for each pitch
    .with_columns(
        pl.col('avg_vra_miss').mean().over('pitch_name').alias('league_avg_vra'),
        pl.col('avg_hra_miss').mean().over('pitch_name').alias('league_avg_hra')
    )
    # abv avg
    .with_columns(
        (pl.col('league_avg_vra') - pl.col('avg_vra_miss')).alias('vra_aa'),
        (pl.col('league_avg_hra') - pl.col('avg_hra_miss')).alias('hra_aa')
    )
    # simple sum of command
    .with_columns(
        (pl.col('vra_aa') + pl.col('hra_aa')).alias('composite_score')
    )
    .sort('composite_score', descending=True) 
)
final_grades.write_csv('grades.csv')
