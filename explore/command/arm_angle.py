# %% arm angle bucketing
import polars as pl
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

df = (
    pl.scan_csv("cleaned_data/pitch_ft_2326.csv").select(["arm_angle", "pitcher_name", "game_year", "pitch_name"])
    .drop_nulls()
)
avg_arm = df.group_by(['pitcher_name', 'pitch_name', 'game_year']).agg(
    aa = pl.col("arm_angle").mean()
).collect(engine="streaming")
avg_arm.head()

# %% plotting
sns.kdeplot(avg_arm["aa"], fill=True)
plt.show()
# %% clustering into groups
X = avg_arm["aa"].to_numpy().reshape(-1, 1)
kmeans = KMeans(n_clusters=5, random_state=26, n_init=100)
labels = kmeans.fit_predict(X)
centers = np.sort(kmeans.cluster_centers_.flatten())
boundaries = (centers[1:] + centers[:-1]) / 2
full_buckets = np.concatenate(([X.min()], boundaries, [X.max()]))

print("bucket ranges:", full_buckets)
