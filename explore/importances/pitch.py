# %% packages import 
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, train_test_split
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.read_parquet('cleaned_data/embed/pitch.parquet')
df = df.sample(n=2000000, shuffle=True)
print(df.columns)

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
        "subsample": np.linspace(0.3, 1, 9),
        "colsample_bytree": [0.5, .65, 0.75, .9, 1],
        "n_estimators": np.linspace(100, 10000, 100, dtype=int),
        "reg_lambda": [1, 3, 5, 10, 20, 25, 35],
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
        n_iter=50,
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
    print(best_params)
    return best_params

# %% train features
features = ['hra', 'vra', 'effective_speed', 'arm_angle', 'release_height',
            'release_x', 'deltax', 'deltaz', 'ay']
df_t = df.drop_nulls()
print(df_t.height)
# %% feature importances
X = df_t.select(features)
y = df_t.select(['pitch_value'])
best_params = train(X, y, 26)
print(best_params)

# %% importances
results = {key: [] for key in features}
for i in range(20):
    print(f'{i} round')
    seed = np.random.randint(0, 1000)
    X = df_t.select(features)
    y = df_t.select(['pitch_value'])
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )
    x_val, x_val2, y_val, y_val2 = train_test_split(
        x_val, y_val, test_size=0.5, random_state=seed, shuffle=True
    )
    # model with random seed
    fmodel = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cpu",
        random_state=seed,
        early_stopping_rounds=30,
        n_jobs=4,
        **best_params,
    )
    fit_params_xgb = {
        "eval_set": [(x_val, y_val)],
        "verbose": False,
    }
    fmodel.fit(X, y, **fit_params_xgb)
    # shap values
    tree = shap.TreeExplainer(fmodel)
    shap_values = tree.shap_values(x_val2)
    # find contribution to each group
    z_exact_values = np.abs(shap_values).mean(axis=0)
    for i in range(len(features)):
        feature = features[i]
        results[feature].append(z_exact_values[i])

# percentages for group
mean_dict = {f: np.mean(results[f]) for f in features}
total_sum = sum(mean_dict.values())
ft_percentages = {f: mean_dict[f]/total_sum for f in features}
print(ft_percentages)