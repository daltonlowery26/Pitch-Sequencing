# %% packages
import polars as pl
import numpy as np
import os
import pymc as pm
from sklearn.kernel_approximation import RBFSampler

os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.read_parquet('cleaned_data/embed/output/pitch_embeded.parquet')
print(df.height)

# %% random fourier features for linear pitch embeddings, decompse embeddings into waves
def rff_basis(pitch_embeddings, n_components=100, length_scale=1.0, random_state=42):
    # width of bell curve for fourier transform
    gamma = 1.0 / (2.0 * length_scale ** 2)
    
    # create random fourier transfromer 
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
    
    # create x basis based on pitch embeddings
    X_basis = rff.fit_transform(pitch_embeddings)
    
    return X_basis

# what do we expect swing traits to see given a pitch
# this is needed because the swing we observe when a batter chooses to swing is just one object of the 
# "batter swing" random variable. this is noisy and to be rigirous we must model a general swing given 
# the pitch
def swing_model(xbasis, y, batter_idx, n_batters, n_traits, n_basis=100):
    model = pm.Model()
    with model:
        # batter index
        b_idx_data = pm.Data("batter_idx_data", batter_idx)
        x_basis = pm.data("X_basis", xbasis)
        # global priors for a swing
        mu_global = pm.Normal("mu_global", mu=0, sigma=1, shape=n_traits)
        # varience between batters, more means we expect greater average diffrences
        sigma_batter = pm.Exponential("sigma_batter", 1.0)
        # offset our expected value (mu) based on the priors of individual batters
        mu_b_offset = pm.Normal("mu_b_offset", mu=0, sigma=1, shape=(n_batters, n_traits))
        mu_b = pm.Deterministic("mu_b", mu_global + (mu_b_offset * sigma_batter))
        # the physics of a swing are consitent across batters, how do we expect "average" swing
        # traits to change based on the pitch embedding. we cannot model this with indiv batters because of
        # data sparsity
        beta_shared = pm.Normal("beta_shared", mu=0, sigma=1, shape=(n_basis, n_traits)) # weights for basis functions
        # expected output of a swing
        # x_basis is the expansion of pitch embeddings into non-linear features
        physics_effect = pm.math.dot(x_basis, beta_shared) 
        mu_y = mu_b[b_idx_data] + physics_effect # average batter swings + what we expect chnages to be based on pitch location
        # create a valid covarience matrices for all the traits
        chol, corr, stds = pm.LKJCholeskyCov("chol_cov", n=n_traits, eta=2.0, sd_dist=pm.Exponential.dist(1.0))
        # likelyhood mdodel
        obs = pm.MvNormal("obs", mu=mu_y, chol=chol, observed=y)
        
    return model

# %% swing traits, only select when all swing traits are there
swing_features = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']

# preparing df
df_s = df.drop_nulls(subset=swing_features)
df_s = df_s.with_columns(
    uYearID = pl.col('batter_id').cast(pl.String) + "_" + pl.col('game_year').cast(pl.String)
)
df_s = df_s.drop_nulls(subset=['uYearID'])
# cast to numeric for pymc
df_s = df_s.with_columns(uYearID_idx = pl.col('uYearID').cast(pl.Categorical).to_physical())
# traits batter idx and num of batters
traits = df_s[swing_features]
b_idx = df_s['uYearID_idx']
num_batters = b_idx.unique().count()

# %% model fit and predict
xBasis = rff_basis(df_s['embeds'])
m = swing_model(xbasis=xBasis, y=traits, batter_idx=b_idx, n_batters = num_batters, n_traits=5)

# %% model fit
with m:
    trace = pm.sample(3000, tune=1000)

# %% values
# predicted mu values
mu_samples = trace.posterior["mu_y"]
print(mu_samples)

# ev for every swing
predicted_mu = mu_samples.mean(dim=["chain", "draw"])
print(predicted_mu)
