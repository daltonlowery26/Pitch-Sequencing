# pitcher try to hit a few targets with each pitch, can we measure the varbility of when they try to hit
# individual targets. based on the situation can we find an estimate of the location they are trying to hit
# or at least some probablity they are trying to hit a location. can we then find how often they deviate from that 
# location. 

# %% hra, vra and command
import polars as pl
import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
    .select(['pitcher_name', 'pitcher_id', 'game_year', 'b_stand', 'count', 'release_height', 'release_x',
        'arm_angle', 'release_extension','p_throws', 'plate_x', 'plate_z', 'hra', 'vra', 'pitch_name'])
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
def bio_mechanical_variance(pitchers, pitches, df=df):
    # intended target cols
    feature_cols = ['release_x', 'release_height', 'vra', 'hra']
    
    # filter valid pitches
    cohort = df.filter(
        (pl.col('pitcher_name').is_in(pitchers)) & 
        (pl.col('pitch_name').is_in(pitches)) &
        (pl.col('game_year') > 2025)
    ).drop_nulls(subset=feature_cols)
    
    cov_lookup = {}
    for keys, group in cohort.group_by(['p_throws', 'pitch_name', 'b_stand']):
        hand, pitch_type, b_stand = keys
        # data 
        data_matrix = group.select(feature_cols).to_numpy()
        # create covariance matrix
        cov_matrix = np.cov(data_matrix, rowvar=False)
        inv_cov = np.linalg.inv(cov_matrix)
        cov_lookup[(hand, pitch_type, b_stand)] = inv_cov
    
    # throws
    p_throws_df = cohort.select(['pitcher_name', 'p_throws']).unique(subset=['pitcher_name'], keep='first')
    p_throws = dict(zip(p_throws_df['pitcher_name'], p_throws_df['p_throws']))
    
    misses_accumulator = []
    
    # looping through
    for keys, indiv_pitch in cohort.group_by(['pitcher_name', 'pitch_name', 'b_stand']):
        name, pitch_type, b_stand, _ = keys
        if len(indiv_pitch) < 10: 
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
        inv_covs = cov_lookup.get((throws, pitch_type, b_stand))

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
avg = command.group_by(['pitcher_name', 'pitch_name']).agg(
    (pl.col(['command_score']).sum() / pl.len()).alias('avg_command'),
    (pl.col(['command_score']).std()).alias('std_command'),
    pl.len().alias('count')
)

# %% pitch value
pitch_grades = (pl.scan_csv('raw_data/indiv_pitch_value25.csv')
                .select(['pitch_name', 'last_name, first_name', 'run_value_per_100', 'est_woba'])
).collect(engine="streaming")
avg = avg.join(pitch_grades, right_on=['last_name, first_name', 'pitch_name'], left_on=['pitcher_name', 'pitch_name'], how='left')

# %% correlations
correlation = (avg
                .filter(pl.col('pitch_name') == '4-Seam Fastball')
                .filter(pl.col('count') > 400)
                .select(pl.corr("avg_command", "run_value_per_100")))
print(correlation.item())
correlation = (avg
                .filter(pl.col('pitch_name') == 'Sinker')
                .filter(pl.col('count') > 50)
                .select(pl.corr("std_command", "avg_command")))
print(correlation.item())
