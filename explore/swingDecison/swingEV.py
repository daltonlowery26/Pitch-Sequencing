# %% packages
import polars as pl
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")
df = pl.read_parquet('cleaned_data/embed/output/pitch_embeded.parquet')

# %% df traits
df_s = df.filter(pl.col('swing'))
print(df_s['description'].unique())
df_s = df_s.with_columns(
    contact = (pl.col('description') == 'hit_into_play').cast(pl.Int16)
)
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
        
class contactLoader(Dataset):
    def __init__(self, df):
        # cols to tensor
        def col_to_tensor(col_name):
            return torch.tensor(df[col_name].to_list(), dtype=torch.float32)
        # extract from df
        pitch_embeddings = col_to_tensor('embeds') # already normalized
        swing_traits = col_to_tensor('traits')
        # zscore of swing traits
        mean = swing_traits.mean(dim=0)
        std = swing_traits.std(dim=0)
        swing_traits = (swing_traits - mean) / std # zscore
        # combined features
        feat = torch.cat([pitch_embeddings, swing_traits], dim=1)
        self.var = feat
        
    def __len__(self):
        return len(self.var)
        
    def __getitem__(self, idx):
        return self.var[idx]


# %% training

        
# %% P(single, double, triple, hr |contact, swing traits, pitch embedding)
