# %% triplet loss, siamese network for indviudal pitches, similarity of pitcher aresenals
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from sklearn.neighbors import NearestNeighbors

# %% knn for similarity
nn_x = pl.scan_csv('cleaned_data/nn_x.csv')
pitch_neigh = NearestNeighbors(metric="euclidean").fit(X=nn_x)

