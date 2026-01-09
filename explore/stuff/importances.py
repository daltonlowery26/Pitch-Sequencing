# %% importances of traits for run prevention
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, train_test_split
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.scan_csv('cleaned_data/pitch_ft_2326.csv').collect(engine="streaming")
df_s = df.sample(n=100000, shuffle=True)

# %% player df
target_cols = [
    'release_extension', 'release_height', 'release_x', 
    'ihb', 'ivb', 'vra', 'hra', 'haa', 'vaa', 
    'release_speed', 'pitch_value', 'arm_angle'
]

df_p = df.group_by(['pitcher_name', 'pitcher_id', 'pitch_name', 'game_year']).agg(
    pl.col(target_cols).mean()
)

# %% command merge
command = pl.read_csv('cleaned_data/player_cmd_grades.csv')
df_p = df_p.join(command, on=['pitcher_id', 'pitcher_name', 'pitch_name', 'game_year'], how='left')

# %% xgb training loop
def train(X, y, seed):
    # train test split
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )
    x_val, x_val2, y_val, y_val2 = train_test_split(
        x_val, y_val, test_size=0.5, random_state=seed, shuffle=True
    )
    # general space to search
    rnd_search_params = {
        "learning_rate": [0.01, 0.001, 0.1],
        "max_depth": np.linspace(2, 10, 5, dtype=int),
        "min_child_weight": np.linspace(1, 40, 10, dtype=int),
        "subsample": np.linspace(0.5, 1, 8),
        "colsample_bytree": [0.5, .65, 0.75, .9, 1],
        "n_estimators": np.linspace(100, 10000, 100, dtype=int),
        "reg_lambda": [1, 3, 5, 10, 20, 25, 35, 45, 55, 70],
    }

    # predefinied split for early stopping
    x_combined = pl.concat([x_train, x_val2])
    y_combined = pl.concat([y_train, y_val2])
    split_index = [-1] * len(x_train) + [0] * len(x_val2)
    pds = PredefinedSplit(test_fold=split_index)
    # reg model
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        device="cpu",
        random_state=seed,
        early_stopping_rounds=30,
        n_jobs=4,
    )
    # random search
    rnd_searcher = RandomizedSearchCV(
        model,
        param_distributions=rnd_search_params,
        cv=pds,
        scoring="neg_root_mean_squared_error",
        n_iter=25,
        random_state=seed,
        verbose=1,
        n_jobs=4,
    )
    # eval set, not verbose
    fit_params_xgb = {
        "eval_set": [(x_val, y_val)],
        "verbose": False,
    }

    # search and extract best params
    search = rnd_searcher.fit(x_combined, y_combined, **fit_params_xgb)
    best_params = search.best_params_
    #print(best_params)
    fmodel = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cpu",
        random_state=seed,
        early_stopping_rounds=30,
        n_jobs=4,
        **best_params,
    )
    # fit model on best params, rmse
    fmodel.fit(x_train, y_train, **fit_params_xgb)
    #ypred = fmodel.predict(x_val2)
    #print(f"RMSE: {root_mean_squared_error(y_true=y_val2, y_pred=ypred)}")
    return fmodel, x_val2, y_val2

# %% train features
df_t = df_p.drop_nulls(subset=['release_extension', 'release_height', 'release_x', 
                'ihb', 'ivb', 'vra', 'hra', 'haa', 'vaa', 'release_speed', 'avg_command', 'arm_angle'])

# %% feature importances
# results
features = ['release_extension', 'release_height', 'release_x', 
                'ihb', 'ivb', 'vra', 'hra', 'haa', 'vaa', 'release_speed', 'avg_command', 'arm_angle']
results = {key: [] for key in features}
for i in range(20):
    X = df_t.select(features)
    y = df_t.select(['pitch_value'])
    seed = np.random.randint(0, 1000)
    # train model with random seed
    model, x_test, y_test = train(X, y, seed=seed)
    # shap values
    tree = shap.TreeExplainer(model)
    shap_values = tree.shap_values(x_test)
    # find contribution to each group
    z_exact_values = np.abs(shap_values).mean(axis=0)
    for i in range(len(features)):
        feature = features[i]
        results[feature].append(z_exact_values[i])

# %% contribution to run value
indices = {
    'release': [0, 1, 2, 11],
    'pitch': [3, 4, 5, 6, 7, 8, 9],
    'command': [10]}
# percentages for group
mean_dict = {f: np.mean(results[f]) for f in features}
total_sum = sum(mean_dict.values())
ft_percentages = {f: mean_dict[f]/total_sum for f in features}
# importances of catagories
category_percentages = {}
for category, idx_list in indices.items():
    total_pct = sum(ft_percentages[features[i]] for i in idx_list)
    category_percentages[category] = total_pct
    print(f'{category}: {total_pct:.2%}')

# release: 21.53%
# pitch: 44.11%
# command: 34.35%
