# %% packages
import polars as pl
import numpy as np
import os
import pymc as pm
from sklearn.kernel_approximation import RBFSampler

df = pl.read_parquet('/content/drive/MyDrive/pitch_embeded.parquet')
print(df.height)

# %% random fourier features for linear pitch embeddings, decompse embeddings into waves
def rff_basis(pitch_embeddings, n_components=50, length_scale=1.0, random_state=42):
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
def swing_model(xbasis, y, batter_idx, n_batters, n_traits, n_basis=50):
    # decompose into principle directions to avoid the ridge problem
    Q_np, R_np = np.linalg.qr(xbasis, mode='reduced')
    # inverse R
    r_inv = np.linalg.inv(R_np)

    # cast to float 32 for jax
    Q_data = Q_np.astype(np.float32)
    r_inv = r_inv.astype(np.float32)
    y = y.astype(np.float32)
    model = pm.Model()

    with model:
        # wrap data in pymc
        b_idx_data = pm.Data("batter_idx_data", batter_idx)
        qshared = pm.Data("xbasis_data", Q_data)
        y_data = pm.Data("y_data", y)

        # global priors for a swing
        mu_global = pm.Normal("mu_global", mu=0, sigma=1, shape=n_traits)

        # varience between batters, more means we expect greater average diffrences
        sigma_batter = pm.Exponential("sigma_batter", 1.0)

        # offset our expected value (mu) based on the priors of individual batters
        mu_b_offset = pm.Normal("mu_b_offset", mu=0, sigma=1, shape=(n_batters, n_traits))
        mu_b = pm.Deterministic("mu_b", mu_global + (mu_b_offset * sigma_batter))

        # the physics of a swing are consitent across batters, how do we expect "average" swing
        # traits to change based on the pitch embedding. we cannot model this with indiv batters because of data sparsity
        theta_shared = pm.Normal("theta_shared", mu=0, sigma=1, shape=(n_basis, n_traits))
        physics_effect = pm.math.dot(qshared, theta_shared)
        mu_y = mu_b[b_idx_data] + physics_effect # average batter swings + what we expect chnages to be based on pitch location

        # create a valid covarience matrices for all the traits
        chol, corr, stds = pm.LKJCholeskyCov("chol_cov", n=n_traits, eta=2.0, sd_dist=pm.Exponential.dist(1.0))

        # likelyhood mdodel
        obs = pm.MvNormal("obs", mu=mu_y, chol=chol, observed=y_data)

        # beta
        beta_recovered = pm.Deterministic("beta", pm.math.dot(r_inv, theta_shared))

        # gpu accelerate
        idata = pm.sample(
            draws=800,
            tune=800,
            chains=4,
            nuts_sampler="numpyro",
            nuts_sampler_kwargs={"chain_method": "vectorized"}
        )
    return idata

# %% swing traits, only select when all swing traits are there
swing_features = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
# preparing df
df_s = df.drop_nulls(subset=swing_features)
df_s = df_s.with_columns(
    uYearID = pl.col('batter_id').cast(pl.String) + "_" + pl.col('game_year').cast(pl.String)
)
df_s = df_s.drop_nulls(subset=['uYearID'])
df_s = df_s.with_columns(
    (pl.col(swing_features) - pl.col(swing_features).mean())/ pl.col(swing_features).std())
# traits batter idx and num of batters
df_s = df_s.with_columns(uYearID_idx = pl.col('uYearID').cast(pl.Categorical).to_physical())
# traits batter idx and num of batters
traits = df_s[swing_features].to_numpy()
b_idx = df_s['uYearID_idx']
num_batters = b_idx.unique().count()

# %% model fit and predict
xBasis = rff_basis(df_s['embeds'])
m = swing_model(xbasis=xBasis, y=traits, batter_idx=b_idx, n_batters = num_batters, n_traits=5) 
m.to_netcdf("/content/drive/MyDrive/model_trace.nc")

# %% values
def add_predicted_values(idata, df, x_basis_cols):
    # weights of fouier featires
    beta_mean = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
    # batter mean
    mu_b_mean = idata.posterior["mu_b"].mean(dim=["chain", "draw"]).values
    # xBasis
    X_mat = df.select(x_basis_cols).to_numpy().astype(np.float32)
    b_idx = df["uYearID_idx"].to_numpy()
    # physics effects
    physics_effect = np.dot(X_mat, beta_mean)
    # batter effect
    batter_effect = mu_b_mean[b_idx]

    # sum compnents
    predicted_matrix = physics_effect + batter_effect

    # predicted values
    return df.with_columns(
        pl.Series("predicted_mu", predicted_matrix)
    )

# %% predicted values
xBasis_pl = pl.DataFrame(xBasis)
predicted_values = pl.concat([xBasis_pl, df_s.select('uYearID_idx'), df_s.select('batter_name'), df_s.select('game_year')], how='horizontal')
final_df = add_predicted_values(m, predicted_values, xBasis_pl.columns)
fea = final_df.select(
    ['batter_name', 'game_year', 'predicted_mu']
)
print(fea.height)

# %% get predicted value
['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
fea = fea.with_columns(
   bat_speed = pl.col('predicted_mu').arr.get(0),
   swing_length = pl.col('predicted_mu').arr.get(1),
   swing_path_tilt = pl.col('predicted_mu').arr.get(2),
   attack_angle = pl.col('predicted_mu').arr.get(3),
   attack_direction = pl.col('predicted_mu').arr.get(4)
)

# %% reverse zscore
og_df = df.drop_nulls(subset=swing_features)
for cols in swing_features:
  fea = fea.with_columns(
      pl.col(cols) * og_df[cols].std() + og_df[cols].mean())

# %% save predicted for pitch
fea.write_parquet('/content/drive/MyDrive/pitch_embeded_predicted.parquet')
