# %% knn for pitches, compare distributions of trajectories for a hitter
import polars as pl
import numpy as np
import scipy
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.scan_csv('cleaned_data/pitch_ft_2326.csv').select(['pitcher_name', 'pitch_name', 'game_year', 
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_extension', 'release_height', 'release_x']).collect(engine="streaming")

# %% disturbtion and comparison
def kinematic_params(df_pitcher):
    # kinematic features
    features = ['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az']
    data = df_pitcher[features].to_numpy()
    
    # mean and covar
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    
    return mu, sigma

def release_params(df_pitcher):
    # release features
    features = ['release_extension', 'release_height', 'release_x']
    data = df_pitcher[features].to_numpy()
    
    # mean and covar
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    
    return mu, sigma

# %% release and kinematic (sometimes simplicty is better than the ai rabbithole)
data = []
for key, pitches in df.group_by(['pitcher_name', 'pitch_name', 'game_year']):
    kmu, ksigma = kinematic_params(pitches)
    rmu, rsigma = release_params(pitches)
    if len(pitches) < 10:
        continue
    
    data.append({
        'pitcher_name': key[0], 'pitch_name': key[1], 'game_year': key[2],
        'kmu': kmu, 'ksigma': ksigma, 'rmu': rmu, 'rsigma': rsigma
    })

# %% compare pitchers
def wasserstein_distance(mu_A, sigma_A, mu_B, sigma_B):
    # distance between means (means of dist)
    diff_mu = np.sum((mu_A - mu_B)**2)
    
    # shape of varience
    sqrt_sigma_A = scipy.linalg.sqrtm(sigma_A)
    
    # geometric mean of two matrices
    term = sqrt_sigma_A @ sigma_B @ sqrt_sigma_A
    sqrt_term = scipy.linalg.sqrtm(term) # varience of overlapping matrix
    
    # floating point errors
    if np.iscomplexobj(sqrt_term):
        sqrt_term = sqrt_term.real
    # take trace of covar matrices to find combined varaince
    trace_term = np.trace(sigma_A + sigma_B - 2 * sqrt_term)
    
    # diffrences in means + diffrence in varince
    w2_sq = diff_mu + trace_term
    
    return np.sqrt(max(0, w2_sq))

