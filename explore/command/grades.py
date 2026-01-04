# %% hra, vra and command
import polars as pl
import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .select(['pitcher_name', 'pitcher_id', 'game_year', 'b_stand', 'count', 'release_height', 'release_x',
        'arm_angle', 'release_extension','p_throws', 'plate_x', 'plate_z', 'hra', 'vra', 'pitch_name'])
    .drop_nulls(subset=['hra', 'vra', 'arm_angle'])
).collect(engine="streaming")

# is a pitcher ahead or behind in the count 
ahead_in_count = ['0-1', '0-2', '1-2', '2-2']
df = df.with_columns(
    ahead = pl.col('count').is_in(ahead_in_count)
)

# arm angle buckets
avg_arm = df.group_by(['pitcher_name', 'pitch_name', 'game_year']).agg(
    aa = pl.col("arm_angle").mean()
)
# based on knn clustering
boundaries = [-65.33636371, 10.33327117, 27.56592316, 
                38.37378127, 48.78640247, 76.30000005]
breaks = boundaries[1:-1]
labels = [str(i) for i in range(1, len(boundaries))]
avg_arm = avg_arm.with_columns(
    arm_angle_bucket = pl.col("aa").cut(breaks, labels=labels).cast(pl.Int64)
)
avg_arm = avg_arm.select(pl.col(['pitcher_name', 'pitch_name', 'game_year', 'arm_angle_bucket']))
# join 
df = df.join(avg_arm, on=['pitcher_name', 'pitch_name', 'game_year'], how='left')

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
def bio_mechanical_variance(pitchers, pitches, df=df):
    # intended target cols
    feature_cols = ['release_x', 'release_height', 'vra', 'hra']
    
    # filter valid pitches
    cohort = df.filter(
        (pl.col('pitcher_name').is_in(pitchers)) & 
        (pl.col('pitch_name').is_in(pitches))
    ).drop_nulls(subset=feature_cols)
    
    cov_lookup = {}
    for keys, group in cohort.group_by(['p_throws', 'pitch_name', 'b_stand', 'game_year']):
        if len(group) < 2: 
            continue
        hand, pitch_type, b_stand, year = keys
        # data 
        data_matrix = group.select(feature_cols).to_numpy()
        # create covariance matrix
        cov_matrix = np.cov(data_matrix, rowvar=False)
        inv_cov = np.linalg.pinv(cov_matrix)
        cov_lookup[(hand, pitch_type, b_stand, year)] = inv_cov
    
    # throws
    p_throws_df = cohort.select(['pitcher_name', 'p_throws']).unique(subset=['pitcher_name'], keep='first')
    p_throws = dict(zip(p_throws_df['pitcher_name'], p_throws_df['p_throws']))
    
    misses_accumulator = []
    
    # looping through
    for keys, indiv_pitch in cohort.group_by(['pitcher_name', 'pitch_name', 'b_stand', 'game_year']):
        name, pitch_type, b_stand, year = keys
        if len(indiv_pitch) < 2: 
            continue
            
        # fit gmm on biomecchanical
        joint_data = indiv_pitch.select(feature_cols).to_numpy()
        
        # git gmm on joint space, learn covariance matrix
        gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=26)
        gmm.fit(joint_data)
        
        # which cluster does a pitch belong too
        probs = gmm.predict_proba(joint_data) 
        predicted_components = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        
        # mean and covar for a cluster
        assigned_means = gmm.means_[predicted_components] 
        
        # get inverse vector based on p_throws and pitch type
        throws = p_throws.get(name)
        inv_covs = cov_lookup.get((throws, pitch_type, b_stand, year))

        # mahoabalhis dist
        weights = np.array([.36648363, .23062016, .46066925, .48497144])   
        diff = (joint_data - assigned_means) * weights
        mahal_sq = np.einsum('ij,jk,ik->i', diff, inv_covs, diff)
        mahal_dist = np.sqrt(mahal_sq)
        
        # 0-1 usi chi sqaured
        prob_metric = chi2.sf(mahal_sq, df=len(feature_cols))
        
        # dist
        pitch_misses = indiv_pitch.with_columns(
            pl.Series("target_confidence", confidence),
            pl.Series("joint_mahal_dist", mahal_dist),  
            pl.Series("command_score", prob_metric)    
        )

        misses_accumulator.append(pitch_misses)
        
    all_misses = pl.concat(misses_accumulator)
    return all_misses

# %% command
pitchers = df['pitcher_name'].unique()
pitches = df['pitch_name'].unique()
command = bio_mechanical_variance(pitchers, pitches)

# %% extracting command value
avg = command.group_by(['pitcher_id', 'pitcher_name', 'pitch_name', 'game_year']).agg(
    pl.col('command_score').mean().alias('avg_command'),
    pl.col('command_score').std().alias('std_command'),
    pl.len().alias('count')
)
avg.write_csv('cmd_grades.csv')
avg = avg.filter(pl.col('game_year') == 2025).filter(pl.col('count') > 100)
# %% year over year stability
yoy_stability = (
    avg.join(
        avg.with_columns((pl.col('game_year') + 1).alias('next_year')),
        left_on=['pitcher_id', 'pitch_name', 'game_year'],
        right_on=['pitcher_id', 'pitch_name', 'next_year'],
        suffix='_prev'
    )
    .group_by('pitch_name')
    .agg(
        pl.corr('avg_command', 'avg_command_prev').alias('yoy_correlation'),
        pl.len().alias('pair_count')
    )
)
print(yoy_stability)

# %% pitch value
pitch_grades = pl.read_csv('raw_data/pitch_values.csv')
cmd_stats = pl.read_csv('raw_data/bot_cmd.csv')
avg = avg.join(pitch_grades, right_on=['MLBAMID'], left_on=['pitcher_id'], how='left')
avg = avg.join(cmd_stats, right_on=['MLBAMID'], left_on=['pitcher_id'], how='left')
print(avg.columns)
# %% std and avg combined
# pca
scaler = StandardScaler()
X_scaled = scaler.fit_transform(avg[['avg_command', 'std_command']])

# pca to combine metrics to one laten space
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled)

# weights and explained var
weights = pca.components_[0]
explained_var = pca.explained_variance_ratio_[0]
print(weights)
avg = avg.with_columns(
    combined = (0.707 * pl.col("avg_command")) + (-0.707 * pl.col("std_command"))
)
# %% correlations
command = ['avg_command', 'botCmd FA', 'wFA/C']
correlation_df = (avg
    .filter(pl.col('pitch_name') == '4-Seam Fastball')
    .select([pl.corr("avg_command", col).alias(col) 
        for col in command])
)
print(correlation_df)


