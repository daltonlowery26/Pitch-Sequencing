# proof of concept
# %% knn for pitches, compare distributions
import polars as pl
import numpy as np
import scipy
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.scan_csv('cleaned_data/pitch_ft_2326.csv').select(['pitcher_name', 'pitcher_id', 'p_throws', 'pitch_name', 'game_year', 
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_extension', 'release_height', 'release_x']).collect(engine="streaming")


# %% disturbtion and comparison
def kinematic_params(df_pitcher):
    # kinematic features
    features = ['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']
    data = df_pitcher[features].to_numpy()
    
    # mean and covar
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    
    return mu, sigma.flatten()

def release_params(df_pitcher):
    # release features
    features = ['release_extension', 'release_height', 'release_x']
    data = df_pitcher[features].to_numpy()
    
    # mean and covar
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    
    return mu, sigma.flatten()

# %% release and kinematic
data = []
for key, pitches in df.group_by(['pitcher_name', 'pitch_name', 'game_year', 'pitcher_id', 'p_throws']):
    if len(pitches) < 20:
        continue
    kmu, ksigma = kinematic_params(pitches)
    rmu, rsigma = release_params(pitches)
    # check for nas
    params = [kmu, ksigma, rmu, rsigma]   
    if any(np.isnan(p).any() or np.isinf(p).any() for p in params):
        continue
    
    # append data 
    data.append({
        'pitcher_name': key[0], 'pitcher_id':key[3], 'pitch_name': key[1], 'p_throws': key[4], 'game_year': key[2],
        'kmu': kmu, 'ksigma': ksigma, 'rmu': rmu, 'rsigma': rsigma
    })

mu_cov = pl.DataFrame(data)
print(mu_cov.height)

# %% add command
cmd = pl.read_csv('cleaned_data/metrics/cmd_grades.csv')
mu_cov = mu_cov.join(cmd, on=['pitcher_name', 'pitch_name', 'pitcher_id', 'game_year'], how='left')
mu_cov = mu_cov.drop_nulls()
mu_cov.head()

# %% add describing info
print(mu_cov.height)
mu_cov.write_parquet('cleaned_data/metrics/pitch_mu_cov.parquet')

# %% compare pitchers
def get_wasserstein_components(mu_A, sigma_A, mu_B_all, sigma_B_all):
    # diffrence between all means
    diff_mu_sq = np.sum((mu_B_all - mu_A)**2, axis=1)
    # varience of A
    sqrt_sigma_A = scipy.linalg.sqrtm(sigma_A)
    
    # only if return complex number
    if np.iscomplexobj(sqrt_sigma_A):
        sqrt_sigma_A = sqrt_sigma_A.real
        
    # find similairty in variences foor all matrices B
    trace_dists = []
    tr_sigma_A = np.trace(sigma_A)
    for sigma_B in sigma_B_all:
        term = sqrt_sigma_A @ sigma_B @ sqrt_sigma_A
        # sqrt_term
        sqrt_term = scipy.linalg.sqrtm(term)
        if np.iscomplexobj(sqrt_term):
            sqrt_term = sqrt_term.real
        # find similairty in varience subtracing the cov
        val = tr_sigma_A + np.trace(sigma_B) - 2 * np.trace(sqrt_term)
        trace_dists.append(val)       
    trace_term = np.array(trace_dists)

    # distance in means and distance in varience 
    w2_sq = diff_mu_sq + trace_term
    return np.sqrt(np.abs(w2_sq))

def nearest_neighbors(target_df, comparison_df, top_n):
    # all comparison data precomputed
    comp_k_mu = np.stack(comparison_df['kmu'].to_numpy())
    comp_k_sigma = np.stack(comparison_df['ksigma'].to_numpy())
    
    comp_r_mu = np.stack(comparison_df['rmu'].to_numpy())
    comp_r_sigma = np.stack(comparison_df['rsigma'].to_numpy())
    
    comp_avg_cmd = comparison_df['avg_command'].to_numpy()
    comp_std_cmd = comparison_df['std_command'].to_numpy()
    
    results_list = []

    # iterate through all target rows
    for target in target_df.iter_rows(named=True):
        # wasserstein distance
        k_dists = get_wasserstein_components(
            target['kmu'], target['ksigma'], comp_k_mu, comp_k_sigma
        )
        
        r_dists = get_wasserstein_components(
            target['rmu'], target['rsigma'], comp_r_mu, comp_r_sigma
        )
        
        # eucledian for scaler
        cmd_dists = np.sqrt(
            (comp_avg_cmd - target['avg_command'])**2 + 
            (comp_std_cmd - target['std_command'])**2
        )

        # normalize so we can sum as terms equal equal terms
        k_mean = np.mean(k_dists)
        r_mean = np.mean(r_dists)
        cmd_mean = np.mean(cmd_dists)
        k_norm = k_dists / k_mean if k_mean > 0 else k_dists
        r_norm = r_dists / r_mean if r_mean > 0 else r_dists
        cmd_norm = cmd_dists / cmd_mean if cmd_mean > 0 else cmd_dists

        # weighted loss based on feature importance
        total_loss = (k_norm * 0.4411) + (r_norm * 0.2153) + (cmd_norm * 0.3435)

        # efficent slice
        closest_indices_pos = np.argpartition(total_loss, top_n)[:top_n]
        
        # match to og row based on lowest loss 
        best_matches = comparison_df[closest_indices_pos].with_columns(
            pl.Series("loss", total_loss[closest_indices_pos])
        ).sort("loss").select(['pitcher_name', 'pitch_name', 'game_year'])

        results = pl.DataFrame({
            'target_pitcher': target['pitcher_name'],
            'target_pitch': target['pitch_name'],
            'target_game_year':target['game_year'],
            'match_pitcher': best_matches['pitcher_name'],
            'match_pitch':best_matches['pitch_name'],
            'match_year':best_matches['game_year']
        })
        # add to results
        results_list.append(results)
        
    return pl.concat(results_list)

# %% nearest neighbors
# need cols and comapison
cols_needed = ['pitcher_name', 'pitch_name', 'game_year', 'p_throws', 'kmu', 'ksigma', 'rmu', 'rsigma', 'avg_command', 'std_command']
mu_cov_compare = mu_cov.select(cols_needed)
mu_cov_search = mu_cov.filter(pl.col('pitcher_name') == 'Skubal, Tarik')
neighbor_data = nearest_neighbors(mu_cov_search, mu_cov_compare, top_n=15)
neighbor_data.write_csv('data.csv')
