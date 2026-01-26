# %% packages
import polars as pl
import os
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.read_parquet()

# %% swing ev
df_s = df.filter(pl.col('swing'))

# %% general res block
class resBlock(nn.Module):
    def __init__(self, dim, dropout):
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

# %% P(contact | swing traits, pitch embedding)
class contactMLP(nn.Module):
    def __init__(self, input, dim, cata, dropout, layers):
        super(contactMLP, self).__init__
        # input
        self.inital = nn.Linear(input, dim)
        # res block
        self.m = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(layers)
        ])
        # post act function
        self.act = nn.BatchNorm1d(dim)
        # non linear
        self.lin = nn.ReLU()
        # to output catagories
        self.out = nn.Linear(dim, cata)
    def forward(self, x):
        x = self.inital(x)
        for layers in self.m:
            x = layers(x)
        x = self.act(x)
        x = self.lin(x)
        x = self.out(x)
        return x
        
class contactLoader(nn.Module):
    def __init__(self, df):
        # cols to tensor
        def col_to_tensor(col_name):
            return torch.tensor(df[col_name].to_list(), dtype=torch.float32)
        
        self.pitch_embeds = df['embeds']
        self.swing_traits = 
        
# %% P(single, double, triple, hr |contact, swing traits, pitch embedding)
