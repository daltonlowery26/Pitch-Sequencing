# find realtive importances of features for location
# %% packages
import polars as pl
import xgboost as xgb 
import numpy as np
from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

df = (pl.scan_csv('cleaned_data/pitch_ft_2326.csv')
        .select(['vra', 'hra', 'release_x', 'arm_angle', 'release_height','plate_x', 'plate_z'])
        .drop_nulls()
).collect(engine="streaming")
print(df.height)
# take random subset
df_s = df.sample(n=200000, shuffle=True)
print(df_s.height)

# %% model training, just need to it be decent, more so trying to extract shap values
def train(X, y):
    # train test split
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=26, shuffle=True) 
    x_val, x_val2, y_val, y_val2 = train_test_split(x_val, y_val, test_size=0.5, random_state=26, shuffle=True) 
    
    # general space to search
    rnd_search_params = {
        'learning_rate': [0.01, 0.001],
        'max_depth': np.linspace(2, 10, 5, dtype=int),
        'min_child_weight': np.linspace(1, 40, 10, dtype=int),
        'subsample': np.linspace(0.5, 0.95, 8),
        'colsample_bytree': [0.5, 0.8, 1],
        'n_estimators': np.linspace(100, 1000, 20, dtype=int),
        'reg_lambda': [1, 3, 5, 10, 20, 25, 35, 45, 55, 70]
    }
    
    # predefinied split for early stopping
    x_combined = pl.concat([x_train, x_val2])
    y_combined = pl.concat([y_train, y_val2])
    split_index = [-1] * len(x_train) + [0] * len(x_val2)
    pds = PredefinedSplit(test_fold=split_index)
    # reg model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method = 'hist',
        device ="cuda",
        random_state=26,
        early_stopping_rounds = 30,
        n_jobs=1
    )
    # random search
    rnd_searcher = RandomizedSearchCV(
        model,
        param_distributions=rnd_search_params,
        cv=pds,
        scoring='neg_root_mean_squared_error',
        n_iter=20,
        random_state=26,
        verbose=1,
        n_jobs=1
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
        objective='reg:squarederror',
        tree_method = 'hist',
        device ="cuda",
        random_state=26,
        early_stopping_rounds = 30,
        n_jobs=1,
        **best_params
    )
    # fit model on best params, rmse
    fmodel.fit(x_train, y_train,  **fit_params_xgb)
    ypred = fmodel.predict(x_val2)
    print(f'RMSE: {root_mean_squared_error(y_true=y_val2, y_pred=ypred)}')
    return fmodel

# xgb models
# %% plate x
plt_x_X = df_s.select(['arm_angle', 'hra', 'release_x', 'release_height'])
plate_x = df_s.select(['plate_x'])
plt_x_model = train(X=plt_x_X, y=plate_x)
xgb.plot_importance(plt_x_model)

# %% plate z
plt_Z_X = df_s.select(['arm_angle', 'vra', 'release_x', 'release_height'])
plate_z = df_s.select(['plate_z'])
plt_z_model = train(X=plt_Z_X, y=plate_z)
xgb.plot_importance(plt_z_model)


