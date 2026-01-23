# create pymc conda

# %% packages
import polars as pl
import numpy as np
import os
import pymc as pm
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.read_parquet('cleaned_data/embed/input/pitch.parquet')

# %% random fourier features for linear pitch embeddings, decompse embeddings into waves
def rff_basis(pitch_embeddings, n_components=50, length_scale=1.0, random_state=42):
    # Gamma is the inverse of the length scale squared
    # It controls the "width" of the bell curve for the kernel
    gamma = 1.0 / (2.0 * length_scale ** 2)
    
    # create random fourier transfromer 
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
    
    # Transform the data
    # Input: (N, d) -> Output: (N, D) where D >> d
    X_basis = rff.fit_transform(pitch_embeddings)
    
    return X_basis

# what do we expect swing traits to see given a pitch
# this is needed because the swing we observe when a batter chooses to swing is just one object of the 
# "batter swing" random variable. this is noisy and to be rigirous we must model a general swing given 
# the pitch
def hierarchical_swing_model(X, y, batter_idx, n_batters, n_traits, n_basis=20):
    with pm.Model() as model:
        # global priors for a swing
        mu_global = pm.Normal("mu_global", mu=0, sigma=1, shape=n_traits)
        # varience between batters, more means we expect greater average diffrences
        sigma_batter = pm.Exponential("sigma_batter", 1.0)
        # offset our expected value (mu) based on the priors of individual batters
        mu_b_offset = pm.Normal("mu_b_offset", mu=0, sigma=1, shape=(n_batters, n_traits))
        mu_b = pm.Deterministic("mu_b", mu_global + mu_b_offset * sigma_batter)
        # the physics of a swing are consitent across batters, how do we expect "average" swing
        # traits to change based on the pitch embedding. we cannot model this with indiv batters because of
        # data sparsity
        beta_shared = pm.Normal("beta_shared", mu=0, sigma=1, shape=(n_basis, n_traits)) # weights for basis functions
        # expected output of a swing
        # x_basis is the expansion of pitch embeddings into non-linear features
        physics_effect = pm.math.dot(X_basis, beta_shared) 
        mu_y = mu_b[batter_idx] + physics_effect # average batter swings + what we expect chnages to be based on pitch location
        # create a valid covarience matrices for all the traits
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol_cov", n=n_traits, eta=2.0, sd_dist=pm.Exponential.dist(1.0)
        )
        # likelyhood mdodel
        obs = pm.MvNormal("obs", mu=mu_y, chol=chol, observed=y)
        
        return model
