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

# %% data loader
class Loader(Dataset):
    def __init__(self, df, features, prob):
        self.df = df
        self.prob = prob
        self.features = df[features].values
        # knn similarity
        self.knn = pitch_neigh
        # intra pitch
        self.pitch_groups = df.group_by(['pitcher_id', 'pitch', 'game_year']).groups
    def __getitem__(self, index):
        anchor_row = self.df.row(index=index)
        anchor = torch.tensor(self.features.row(index=index, dtype=torch.float32))
        # random mix between knn and groups 
        rnd = torch.rand(1)
        # knn
        if rnd < self.prob:
            nearest = self.knn.kneighbors(anchor_row, return_distance=False)
            ind = torch.randint(1, 5, (1,))
            nearest = nearest[f"column_{ind}"]
            positive = self.df.row(index=nearest)
        else:
            # pitcher of same type and from same pitcher
            key = (anchor_row['pitcher_id'], anchor_row['pitch_type'], anchor_row['game_year'])
            # all similar ids
            sim = self.pitch_groups[key]
            # random choice of id
            ind = np.random.choice(sim)
            # features
            positive = torch.tensor(self.features[ind], dtype=torch.float32)            
        
        return anchor, positive, index
# pitch loss
class tripletLoss(nn.Module): # with hard mining
    def __init__(self, anchor, margin):
        super(tripletLoss, self).__init__
        self.margin = margin
    def forward(self, anchors, positives, anchor_types):
        # dist between anchor and pos
        d_pos = torch.norm(anchors-positives, p=2, dim=1)
        # distance matrix
        neg_pool = torch.cat([anchors, positives])
        
