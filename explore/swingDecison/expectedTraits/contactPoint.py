# %% packages
import os
import polars as pl
import numpy as np
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/')

# load and select data
swing_features = ['attack_direction', 'embed']
df = pl.read_parquet('cleaned_data/metrics/xswing/swingTraits.parquet')
swing = pl.scan_csv('cleaned_data/pitch_2015_2026.csv').select(['game_pk', 'batter_id',  'pitcher_id', 'at_bat_number', 'pitch_number', 'count', 'game_date', 'game_year', 'intercept_ball_minus_batter_pos_x_inches',
    'intercept_ball_minus_batter_pos_y_inches']).collect(engine="streaming")
swing = swing.rename({'intercept_ball_minus_batter_pos_x_inches':'intercept_x', 'intercept_ball_minus_batter_pos_y_inches':'intercept_y'})
df = df.join(swing, on=['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count'], validate='1:1' ,how='left')

# %% train loop
def train(X, y, seed):
    # train test split
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, test_size=0.5, random_state=seed, shuffle=True
    )
    # general space to search
    rnd_params = {
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(3, 9),
        "min_child_weight": randint(10, 500),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "gamma": uniform(0, 5),
        "reg_lambda": loguniform(1e-3, 100),
        "reg_alpha": loguniform(1e-3, 100),
        "n_estimators": randint(500, 3000)
    }
    # reg model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        multi_strategy='multi_output_tree',
        tree_method='hist',
        random_state=seed,
        early_stopping_rounds=20,
        eval_metric = 'rmse'
    )
    # random search
    rnd_searcher = RandomizedSearchCV(
        model,
        param_distributions=rnd_params,
        cv=4,
        scoring="neg_log_loss",
        n_iter=50,
        random_state=seed,
        verbose=10,
        n_jobs=-1,
    )
    # eval set, not verbose
    fit_params = {
        "eval_set": [(x_val, y_val)],
        "verbose": True
    }

    # search and extract best params
    search = rnd_searcher.fit(x_train, y_train, **fit_params)
    best_params = search.best_params_
    print(best_params)
    return best_params

# %%
df = df.with_columns(
    pl.col('effective_speed').fill_null(pl.col('release_speed'))
)

# %% train on real swing traits, pitch velocity
features = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction', 'effective_speed', 'intercept_x', 'intercept_y']
df_s = df.drop_nulls(subset=features)
df_s = df_s.with_columns(pl.col(features).cast(pl.Float32))

# x and y
x_act = df_s['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction', 'effective_speed']
y_act = df_s['intercept_x', 'intercept_y']

# train test split
x_train, x_val, y_train, y_val = train_test_split(x_act, y_act, test_size=0.3, random_state=26, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=26, shuffle=True)

# best hyperparmeters from training
best_params = {'colsample_bytree': np.float64(0.6694154832024045), 'gamma': np.float64(1.0190788385067395), 'learning_rate': np.float64(0.017052179128049127), 
    'max_depth': 6, 'min_child_weight': 117, 'reg_alpha': np.float64(0.021815263073415744),
    'reg_lambda': np.float64(20.738000927742434), 'subsample': np.float64(0.7306656624447523)}

# fit model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    multi_strategy='multi_output_tree',
    n_estimators=10000,
    tree_method='hist',
    random_state=26,
    early_stopping_rounds=20,
    eval_metric = 'rmse',
    **best_params
)
fit_params = {
    "eval_set": [(x_val, y_val)]
}
trained = model.fit(x_train, y_train, **fit_params)

# %% predict swings
xValues = ['bat_speed_x', 'swing_length_x', 'swing_path_tilt_x', 'attack_angle_x', 'attack_direction_x', 'effective_speed']
xswing = pl.read_parquet('cleaned_data/metrics/xswing/swingTraits.parquet')
xswing = df_s.with_columns(pl.col(xValues).cast(pl.Float32))
swings = xswing.select(pl.col(xValues))
predicted = trained.predict(swings)

# %%
plPred = pl.DataFrame(predicted).rename({'column_0':'pred_intercept_x', 'column_1':'pred_intercept_y'})
xswingCon = pl.concat([xswing, plPred], how='horizontal')

# %%
xswingCon.write_parquet('cleaned_data/metrics/xswing/xtraitContact.parquet')
