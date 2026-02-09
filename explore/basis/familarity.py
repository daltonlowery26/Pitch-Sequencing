# is familarity pen a product of location similairty or pitch type
# %% packages
import os
import polars as pl
import catboost as cb
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/')

# load and select data
df = (pl.scan_parquet('cleaned_data/embed/output/pitch_umap150.parquet')).collect(engine="streaming")