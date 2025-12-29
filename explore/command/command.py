# pitcher try to hit a few targets with each pitch, can we measure the varbility of when they try to hit
# individual targets. based on the situation can we find an estimate of the location they are trying to hit
# or at least some probablity they are trying to hit a location. can we then find how often they deviate from that 
# location. 
# %% hra, vra and command
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
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

# %% explore individual pitchers
indiv = df.filter((pl.col('pitcher_name') == "Kirby, George") & (pl.col('game_year') == 2025))
print(indiv.select('pitch_name').unique())

# %% plotting pitch loc and intention
def plot_pitch_intent(pitch_data, targets):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # plot pitches
    ax.scatter(pitch_data[:, 0], pitch_data[:, 1], 
               alpha=0.3, color='steelblue', s=15, label='Actual Pitches')
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

# %% expected location given pitcher, pitch, count, batter handedness, probablity of location
indiv_pitch = indiv.filter(
    (pl.col('pitch_name') == '4-Seam Fastball') &  
    (pl.col('b_stand') == 'R') & 
    (pl.col('ahead'))
)
print(indiv_pitch.height)
indiv_loc = indiv_pitch.select(['plate_x', 'plate_z'])
indiv_loc = indiv_loc.to_numpy()
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=26)
gmm.fit(indiv_loc)
intended_locations = gmm.means_

# plotting function
plot_pitch_intent(indiv_loc, intended_locations)

# %% low likelyhood points, obvious misses, score on these
scores = gmm.score_samples(indiv_loc)
threshold = np.percentile(scores, 10)
is_outlier = scores < threshold
outlier_indices = np.where(is_outlier)[0]
pitch_misses = (
    indiv_pitch.with_row_index(name="id").filter(pl.col("id").is_in(outlier_indices)).drop("id")
)
pitch_misses.head()
# %% find expected vra and hra given arm angle, release height, extension and location

# %% diffrence in expect vra and hra from expected location