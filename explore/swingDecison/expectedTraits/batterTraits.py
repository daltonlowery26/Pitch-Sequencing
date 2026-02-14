# %% packages
import os
import polars as pl
import numpy as np
import jax
import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from scipy.spatial.distance import pdist
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import KFold
os.chdir('/Users/daltonlowery/Desktop/projects/Optimal Pitch/data')

# add game date
df = pl.read_parquet('cleaned_data/embed/output/pitch_umap150.parquet')
swing = pl.scan_csv('cleaned_data/pitch_2015_2026.csv').select(['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count', 'game_date', 'game_year']).collect(engine="streaming")
df = df.join(swing, on=['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count'], validate='1:1' ,how='left')

# %% random fourier features for linear pitch embeddings, decompse embeddings into waves
def rff_embeds(embeddings, dim=64):
    # gamma heuristic, take sample embeddings and compute distance 
    rng = np.random.default_rng(26)
    sampleEmbeddings = rng.choice(embeddings, size = 2048, replace = False)
    dist = pdist(sampleEmbeddings, metric='sqeuclidean')
    del sampleEmbeddings
    # median distance, get gamma value
    medianDist = np.median(dist)
    gamma = 1 / (2 * medianDist)
    del dist
    # rff features for embeddings
    rff = RBFSampler(gamma=gamma, n_components=dim, random_state=26)
    rffEmbed = rff.fit_transform(embeddings)
    
    return rffEmbed

# %% E[Swing Traits| Previous Swings, Pitch]
def swing_model(embeddings, swings, batter_idx, n_batters, n_obs, batch_size, n_traits=5, n_basis=64):
    # global expecation
    global_mu = npro.sample("global_mu", dist.Normal(0.0, 1.0).expand([n_traits]))
    
    # varience between batters
    sigma_batters = npro.sample("sigma_batters", dist.Exponential(0.5))
    
    # batter offset from global expecation
    with npro.plate("batters_plate", n_batters):
        mu_offset = npro.sample("mu_offset", dist.StudentT(df=2, loc=0.0, scale=1.0).expand([n_traits]).to_event(1))
    
    # batter expecation
    mu_b = npro.deterministic("mu_b", global_mu + (mu_offset * sigma_batters))
    
    # embedding coeffs
    theta_shared = npro.sample("theta_shared", dist.Normal(0.0, 1.0).expand([n_basis, n_traits]))
    
    # std of each trait
    sigma_traits = npro.sample("sigma_traits", dist.HalfNormal(1.0).expand([n_traits]))
    
    # low triangular
    l_omega = npro.sample("l_omega", dist.LKJCholesky(n_traits, 0.8)) # <1 bc we know some traits are correlated
    
    # covarience matrix 
    l_cov = jnp.matmul(jnp.diag(sigma_traits), l_omega)
    
    # sample
    with npro.plate("obs_plate", n_obs, subsample_size=batch_size) as ind:
        batch_embed = embeddings[ind]
        batch_batter_idx = batter_idx[ind]
        batch_sample = swings[ind] if swings is not None else None
        # mu for batch
        mu_traits = npro.deterministic("mu_traits", mu_b[batch_batter_idx] + jnp.dot(batch_embed, theta_shared))
        npro.sample("obs", dist.MultivariateStudentT(df=5, loc=mu_traits, scale_tril = l_cov), obs=batch_sample)

def trainModel(embeddings,swing_mask, swings, batter_idx, n_batters):
    # append expecation for each swing
    expectations = np.zeros_like(swings, dtype=np.float32)
    
    # all swing indices, mask takes
    all_indices = np.arange(len(swings))
    valid_swing_indices = all_indices[swing_mask]
    take_indices = all_indices[~swing_mask]
    
    # splits
    kf = KFold(n_splits=20, shuffle=True, random_state=26)
    take_splits = np.array_split(take_indices, 20) # split takes into 20 arrays
    
    for fold, (trainIdx, testIdx) in enumerate(kf.split(batter_idx)):
        trainIdx = valid_swing_indices[trainIdx]
        test_swing_idx = valid_swing_indices[testIdx]
        test_take_idx = take_splits[fold]
        pred_idx = np.concatenate([test_swing_idx, test_take_idx]) #
        
        # train folds
        tEmbeddings = jnp.array(embeddings[trainIdx])
        tBatterIdx = jnp.array(batter_idx[trainIdx])
        tSwings = jnp.array(swings[trainIdx])
        tObs = len(tBatterIdx)
        
        # held out fold
        iEmbeddings = jnp.array(embeddings[pred_idx])
        iBatterIdx = jnp.array(batter_idx[pred_idx])
        iObs = len(iBatterIdx)
        
        # model run
        guide = autoguide.AutoNormal(swing_model)
        optimizer = npro.optim.Adam(step_size=0.001)
        svi = SVI(swing_model, guide, optimizer, loss=Trace_ELBO())
        rng_key_train, rng_key_pred = jax.random.split(jax.random.PRNGKey(fold), 2)
                
        svi_result = svi.run(rng_key_train, 15000,
                                embeddings=tEmbeddings,
                                swings=tSwings,
                                batter_idx=tBatterIdx,
                                n_obs=tObs,         
                                n_batters=n_batters, 
                                batch_size=128) 
        
        params = svi_result.params
        predictive = Predictive(model=swing_model,
                                guide=guide,
                                params=params,
                                num_samples=250,
                                return_sites=["mu_traits"]) 
        
        predictions = predictive(rng_key_pred,
                                    embeddings= iEmbeddings,
                                    swings=None,
                                    batter_idx= iBatterIdx,
                                    n_obs=iObs,
                                    n_batters=n_batters,
                                    batch_size = len(iEmbeddings)
        )
        
        foldExp = predictions["mu_traits"].mean(axis=0)
        
        expectations[testIdx] = foldExp
    return expectations

# %% swing traits, preparing df
swing_features = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
df = df.with_columns(uYearID = pl.col('batter_id').cast(pl.String) + "_" + pl.col('game_year').cast(pl.String))
df = df.drop_nulls(subset=['uYearID'])
df = df.sort(['game_date', 'at_bat_number'])
dfTraits = df.select(swing_features)

# zscore
dfTraits = dfTraits.with_columns((pl.col(swing_features) - pl.col(swing_features).mean())/ pl.col(swing_features).std())
is_swing = ~np.isnan(dfTraits).any(axis=1)

# %% traits for model
df = df.with_columns(uYearID_idx = pl.col('uYearID').cast(pl.Categorical).to_physical())
traits = dfTraits.to_numpy()
b_idx = df['uYearID_idx']
num_batters = b_idx.unique().count()

# embedding altering
rffEmbeds = rff_embeds(df['embed'])
q_embeds, r_embeds = np.linalg.qr(rffEmbeds, mode='reduced')

# %% model fit and predict
outCome = trainModel(embeddings=q_embeds, swing_mask = is_swing, swings=traits, batter_idx=b_idx, n_batters=num_batters)

# %%
schema = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
actualizedTraits = pl.DataFrame(outCome, schema=schema)

# inverse zscore
actualizedTraits = actualizedTraits.with_columns(
    bat_speed = pl.col('bat_speed') * df['bat_speed'].std() + df['bat_speed'].mean(),
    swing_length = pl.col('swing_length') * df['swing_length'].std() + df['swing_length'].mean(),
    swing_path_tilt = pl.col('swing_path_tilt') * df['swing_path_tilt'].std() + df['swing_path_tilt'].mean(),
    attack_angle = pl.col('attack_angle') * df['attack_angle'].std() + df['attack_angle'].mean(),
    attack_direction = pl.col('attack_direction') * df['attack_direction'].std() + df['attack_direction'].mean(),
)

# %%
cmb = df.drop_nulls(subset=swing_features)
cmb = cmb.with_columns(uYearID = pl.col('batter_id').cast(pl.String) + "_" + pl.col('game_year').cast(pl.String))
cmb = cmb.drop_nulls(subset=['uYearID'])
cmb = cmb.sort(['game_date', 'at_bat_number'])
actualizedTraits = actualizedTraits.select(pl.all().name.suffix('_x'))
combined = pl.concat([cmb, actualizedTraits], how='horizontal')

# %%
player = combined.filter(pl.col('batter_name') == 'Cruz, Oneil').select(['bat_speed', 'bat_speed_x']).mean()

# %%
combined.write_parquet('cleaned_data/metrics/xswing/swingTraits.parquet')
