# %% packages
import polars as pl
import jax
import jax.numpy as jnp
import numpy as np
import os
import numpyro as npro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from sklearn.kernel_approximation import RBFSampler
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold

# load and select data
swing_features = ['attack_direction', 'embed']
df = (pl.scan_parquet('/content/drive/MyDrive/BNpitchUEmbed.parquet')
    .select(['game_pk', 'batter_id', 'batter_name', 'pitcher_id', 'at_bat_number', 'description', 'pitch_number', 'count', 'bat_speed', 'attack_direction', 'embed'])
    .drop_nulls(subset=swing_features)).collect(engine="streaming")

# %% timing metrics
swing = pl.scan_csv('/content/drive/MyDrive/pitch_2015_2026.csv').select(['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count',
                    'game_year','intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches']).collect(engine="streaming")
df = df.join(swing, on=['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count'], validate='1:1' ,how='left') # validate 1:1 param ensures matches
df = df.drop_nulls()
print(df.height)

# %%
avgPos = df.group_by(['batter_name', 'batter_id', 'game_year']).agg(
    ymean = pl.col('intercept_ball_minus_batter_pos_y_inches').cast(pl.Float32).mean(),
    ystd = pl.col('intercept_ball_minus_batter_pos_y_inches').cast(pl.Float32).std(),
    xmean = pl.col('intercept_ball_minus_batter_pos_x_inches').cast(pl.Float32).mean(),
    xstd = pl.col('intercept_ball_minus_batter_pos_x_inches').cast(pl.Float32).std(),
    foul = (pl.col('description') == 'foul').cast(pl.Float32).mean()
)
print(avgPos.select(pl.corr('foul','ystd')))


def rff_embeds(embeddings, dim=64):
    # gamma heuristic, take sample embeddings and compute distance 
    rng = np.random.default_rng(26)
    sampleEmbeddings = rng.choice(embeddings[0], size = 1000, replace = False)
    dist = pdist(sampleEmbeddings, metric='sqeuclidean')
    # median distance, get gamma value
    medianDist = np.median(dist)
    gamma = 1 / (2 * medianDist)
    
    rff = RBFSampler(gamma=gamma, n_components=64, random_state=26)
    rffEmbed = rff.fit_transform(embeddings)
    
    return rffEmbed

# %% E[meanPos | Pitch Embedding, Batter]
def expectedBallPos(embeddings, batter_indices, num_batters, sample, batch_size):
    n_obs, embedding_dim = embeddings.shape
    outDim = 2
    # coeff for embeddings
    beta = npro.sample("beta", dist.Normal(0, 1).expand([embedding_dim, outDim]).to_event(2))
    # learn varience for each batter
    sigma_batter = npro.sample("sigma_batter", dist.HalfNormal(1.0).expand([outDim]).to_event(1))
    # batter expectation
    with npro.plate("batter_plate", num_batters):
        alpha_raw = npro.sample("alpha_raw", dist.Normal(0,1).expand([outDim]).to_event(1))
        alpha = alpha_raw * sigma_batter

    # expected mu given batter and pitch
    mu = alpha[batter_indices] + jnp.dot(embeddings, beta)

    # covaraince matrix
    sd_obs = npro.sample("sd_obs", dist.HalfNormal(1.0).expand([outDim]).to_event(1))

    # correaltion matrix for varience
    l_omega = npro.sample("L_omega", dist.LKJCholesky(outDim, concentration=1.0))

    # scale correlation by factor of noise
    l_sigma = sd_obs[..., None] * l_omega

    # sample
    nu = npro.sample("nu", dist.Gamma(2.0, 0.1))
    with npro.plate("data", n_obs, subsample_size=batch_size) as ind:
        batch_embed = embeddings[ind]
        batch_batter_idx = batter_indices[ind]
        batch_sample = sample[ind]
        # predict average given batter and pitch
        mu = npro.deterministic("mu", alpha[batch_batter_idx] + jnp.dot(batch_embed, beta))
        npro.sample("obs", dist.MultivariateStudentT(df=nu, loc=mu, scale_tril=l_sigma), obs=batch_sample)


def trainModel(embeddings, swings, batter_idx, n_batters):
    # append expecation for each swing
    expectations = np.zeros_like(swings, dtype=np.float32)
    # splits
    kf = KFold(n_splits=20, shuffle=True, random_state=42)
    
    for fold, (trainIdx, testIdx) in enumerate(kf.split(batter_idx)):
        # train folds
        tEmbeddings = jnp.array(embeddings[trainIdx])
        tBatterIdx = jnp.array(batter_idx[trainIdx])
        tSwings = jnp.array(swings[trainIdx])
        tObs = len(tBatterIdx)
        
        # held out fold
        iEmbeddings = jnp.array(embeddings[testIdx])
        iBatterIdx = jnp.array(batter_idx[testIdx])
        iObs = len(iBatterIdx)
        
        # model run
        guide = autoguide.AutoNormal(expectedBallPos)
        optimizer = npro.optim.Adam(step_size=0.001)
        svi = SVI(expectedBallPos, guide, optimizer, loss=Trace_ELBO())
        rng_key_train, rng_key_pred = jax.random.split(jax.random.PRNGKey(fold), 2)
                
        svi_result = svi.run(rng_key_train, 10000,
                                embeddings=tEmbeddings,
                                swings=tSwings,
                                batter_idx=tBatterIdx,
                                num_obs=tObs,         
                                n_batters=n_batters, 
                                batch_size=4096) 
        
        params = svi_result.params
        predictive = Predictive(model=expectedBallPos,
                                guide=guide,
                                params=params,
                                num_samples=1000,
                                return_sites=["mu"]) 
        
        predictions = predictive(rng_key_pred,
                                    embeddings=iEmbeddings,
                                    swings=None,
                                    batter_idx=iBatterIdx,
                                    num_obs=iObs,
                                    n_batters=n_batters)
        
        foldExp = predictions["obs"].mean(axis=0)
        
        expectations[testIdx] = foldExp
    return expectations

# %% prepare
df = df.with_columns(
    uYearID = pl.col('batter_id').cast(pl.String) + "_" + pl.col('game_year').cast(pl.String)
)
df = df.with_columns(yID_idx = pl.col('uYearID').cast(pl.Categorical).to_physical())

nBat = df['yID_idx'].unique().count()

# true results
sample = jnp.array(
    df.with_columns(
        pl.col('intercept_ball_minus_batter_pos_x_inches').cast(pl.Float32),
        pl.col('intercept_ball_minus_batter_pos_y_inches').cast(pl.Float32)
    ).select(['intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches'])
    .to_numpy()
)

# batter ids
batID = jnp.array(df['yID_idx'].to_numpy())

# %% embeds
embeds = df['embed'].to_numpy()
rffEmbeddings = rff_embeds(embeds)
q_embeds, r_embeds = np.linalg.qr(rffEmbeddings, mode='reduced')
r_inv = np.linalg.inv(r_embeds)

# %%
outCome = trainModel(embeddings=q_embeds, swings=sample, batter_idx=batID, n_batters=len(np.unique(batID)))

# %% 
schema = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
actualizedTraits = pl.DataFrame(outCome, schema=schema)

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
print(combined)

