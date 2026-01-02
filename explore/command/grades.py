# pitcher try to hit a few targets with each pitch, can we measure the varbility of when they try to hit
# individual targets. based on the situation can we find an estimate of the location they are trying to hit
# or at least some probablity they are trying to hit a location. can we then find how often they deviate from that 
# location. 
# %% hra, vra and command
import polars as pl
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, rankdata
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

# %% plot pitches, intended locations
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
# %% intended target
def intended_target(pitchers, pitches, df=df):
    # filter valid pitches
    cohort = df.filter(
        (pl.col('pitcher_name').is_in(pitchers)) & 
        (pl.col('pitch_name').is_in(pitches)) &
        (pl.col('game_year') == 2025)
    )
    # looping through
    misses_accumulator = []
    for keys, indiv_pitch in cohort.group_by(['pitcher_name', 'pitch_name', 'b_stand', 'ahead']):
        # pitch count
        if len(indiv_pitch) < 50: 
            continue
        # location of pitches
        indiv_loc = indiv_pitch.select(['plate_x', 'plate_z']).to_numpy()
        
        # gmm to find intended location
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=26)
        gmm.fit(indiv_loc)
        probs = gmm.predict_proba(indiv_loc) # Shape (N, 3)
        
        # most likley canadiate
        predicted_components = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        
        # means and cov of predicted value
        means = gmm.means_[predicted_components]      
        covs = gmm.covariances_[predicted_components]
        
        # statistical distance from mean
        diffs = indiv_loc - means 
        inv_covs = np.linalg.inv(covs) 
        mahal_sq = np.einsum('ij,ijk,ik->i', diffs, inv_covs, diffs)
        mahal_dist = np.sqrt(mahal_sq)
        
        # distance by confidence
        weighted_penalty = mahal_sq * confidence
        
        # return all pitches
        is_valid = mahal_dist > -1 
        
        # pitches and cluster
        pitch_misses = indiv_pitch.filter(pl.lit(is_valid))
        pitch_misses = pitch_misses.with_columns(
            pl.Series("target_x", means[is_valid, 0]), # predicted location
            pl.Series("target_z", means[is_valid, 1]), 
            pl.Series("target_confidence", confidence[is_valid]), # confidence it belongs to cluster
            pl.Series("mahalanobis_sq", mahal_sq[is_valid]), # distance 
            pl.Series("weighted_pen", weighted_penalty[is_valid]) # weight
        )

        misses_accumulator.append(pitch_misses)

    all_misses = pl.concat(misses_accumulator)
    return all_misses

# %% command
pitchers = df['pitcher_name'].unique()
pitches = df['pitch_name'].unique()
command = intended_target(pitchers, pitches)

# %% extracting command value
avg = command.group_by(['pitcher_name', 'pitch_name']).agg(
    pl.col(['weighted_pen']).sum() / pl.len()
)
avg.write_csv('ind.csv')
