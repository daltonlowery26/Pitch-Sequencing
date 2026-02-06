# %%
import polars as pl
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# %% model and train blocks
class resBlock(nn.Module):
    def __init__(self, dim, dropout):
        super(resBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class foulModel(Dataset):
    def __init__(self, df, embeddings, timing):
        # helper function
        embeddings = torch.tensor(df['embeds'].to_list()).to('cuda')
        swingTraits = df['traits']
        
        mean = swingTraits.mean(dim=0)
        std = swingTraits.std(dim=0)
        traits = (swingTraits - mean) / std
        zTraits = torch.tensor(traits.to_list()).to('cuda')
        
        self.e = embeddings
        self.s = zTraits
    
    def __len__(self):
        return len(self.e)
    
    # return indices and hold dataset on gpu
    def __getitem__(self, idx):
        return {
            'embed': self.e[idx],
            'swing': self.s[idx]
        }
