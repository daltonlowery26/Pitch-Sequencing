# find realtive importances of features for location
# %% packages
import os
import shap
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, train_test_split
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = (
    pl.scan_csv("cleaned_data/pitch_ft_2326.csv").select(["vra", "hra", "release_x", "arm_angle", "release_height", "plate_x", "plate_z"])
    .drop_nulls()
).collect(engine="streaming")
# take random subset
df_s = df.sample(n=100000, shuffle=True)

# %% model training loop
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
        "n_estimators": np.linspace(100, 1000, 100, dtype=int),
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

# %% single shot xgb model
# plate x
plt_x_X = df_s.select(["arm_angle", "hra", "vra", "release_x", "release_height"])
plate_x = df_s.select(["plate_x"])
plt_x_model, x_test, y_test = train(X=plt_x_X, y=plate_x, seed=26)

# plate z
plt_Z_X = df_s.select(["arm_angle", "hra", "vra", "release_x", "release_height"])
plate_z = df_s.select(["plate_z"])
plt_z_model, x_test, y_test = train(X=plt_Z_X, y=plate_z, seed=26)

# %% plt_x shap values
features = [ "hra", "vra", "release_x", "release_height"]
x_results =  {key: [] for key in features}
for i in range(10):
    plt_x_X = df_s.select(["hra", "vra", "release_x", "release_height"])
    plate_x = df_s.select(["plate_x"])
    seed = np.random.randint(0, 1000)
    # train model with random seed
    plt_x_model, x_test, y_test = train(X=plt_x_X, y=plate_x, seed=seed)
    # shap values
    tree = shap.TreeExplainer(plt_x_model)
    shap_values = tree.shap_values(x_test)
    # find contribution to each group
    z_exact_values = np.abs(shap_values).mean(axis=0)
    # add run to 
    for i in range(len(features)):
        feature = features[i]
        x_results[feature].append(z_exact_values[i])

# %% plt_z shap values
z_results =  {key: [] for key in features}
for i in range(20):
    plt_Z_X = df_s.select(["hra", "vra", "release_x", "release_height"])
    plate_z = df_s.select(["plate_z"])
    seed = np.random.randint(0, 1000)
    # train model with random seed
    plt_z_model, z_test, y_test = train(X=plt_Z_X, y=plate_z, seed=seed)
    # shap values
    tree = shap.TreeExplainer(plt_z_model)
    shap_values = tree.shap_values(z_test)
    # find contribution to each group
    z_exact_values = np.abs(shap_values).mean(axis=0)
    # add run to 
    for i in range(len(features)):
        feature = features[i]
        z_results[feature].append(z_exact_values[i])

# %% feature importance
for feature in features:
    # get raw contirbutions
    raw_x_contributions = np.array(x_results[feature])
    raw_z_contributions = np.array(z_results[feature])
    raw_z_contributions = raw_z_contributions[:10]
    # l2 importance
    magnitude_importance = np.sqrt(raw_x_contributions**2 + raw_z_contributions**2)
    # feature
    print(f"Feature: {feature}")
    print(f"Mean Spatial Shift: {magnitude_importance.mean():.4f} inches")

