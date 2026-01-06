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
print(df.columns)
df_s = df.sample(n=100000, shuffle=True)
# %% xgb training loop
def train(X, y):
    # train test split
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=26, shuffle=True
    )
    x_val, x_val2, y_val, y_val2 = train_test_split(
        x_val, y_val, test_size=0.5, random_state=26, shuffle=True
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
        random_state=26,
        early_stopping_rounds=30,
        n_jobs=4,
    )
    # random search
    rnd_searcher = RandomizedSearchCV(
        model,
        param_distributions=rnd_search_params,
        cv=pds,
        scoring="neg_root_mean_squared_error",
        n_iter=10,
        random_state=26,
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
    fmodel = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cpu",
        random_state=26,
        early_stopping_rounds=30,
        n_jobs=4,
        **best_params,
    )
    # fit model on best params, rmse
    fmodel.fit(x_train, y_train, **fit_params_xgb)
    ypred = fmodel.predict(x_val2)
    print(f"RMSE: {root_mean_squared_error(y_true=y_val2, y_pred=ypred)}")
    return fmodel, x_val2, y_val2

# %% train features
df_s = df_s.drop_nulls(subset=['release_extension', 'release_height', 'arm_angle', 'release_x', 
                'release_spin_rate', 'spin_axis', 'vra', 'hra', 'pitch_value', 'release_speed'])
X = df_s.select(['release_extension', 'release_height', 'arm_angle', 'release_x', 
                'release_spin_rate', 'spin_axis', 'vra', 'hra', 'release_speed'])
y = df_s.select(['pitch_value'])

# %% model train and validation
model, x_test, y_test = train(X, y)
print(r2_score(y_true=y_test, y_pred=model.predict(x_test)))
print(mean_absolute_error(y_true=y_test, y_pred=model.predict(x_test)))

# %% shap values
tree = shap.TreeExplainer(model)
sample = X.sample(10000, shuffle=True)
shap_values = tree.shap_values(sample)
shap.summary_plot(
    shap_values, 
    features=X, 
    feature_names=X.columns, 
    plot_type="bar"
)
z_exact_values = np.abs(shap_values).mean(axis=0)
print(z_exact_values)
