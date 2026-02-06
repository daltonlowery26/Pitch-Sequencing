# %%
import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %% model and train blocks
class resBlock(nn.Module):
    def __init__(self, dim, dropout):
        super(resBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.block(x)

class bipModel(nn.Module):
    def __init__(self, embedding, swingTraits, hidden, layers, outputs):
        # embeddings
        self.eInput = nn.Linear(64, hidden)
        self.eBlock = nn.ModuleList([
            resBlock(hidden, 0.2) for _ in range(layers)
        ])
        # swingTraits
        self.sInput = nn.Linear(5, hidden)
        self.sBlock = nn.ModuleList([
            resBlock(hidden, 0.2) for _ in range(layers)
        ])
        # output
        self.combine = nn.Linear(hidden * 2, hidden)
        self.backBone = nn.ModuleList([
            resBlock(hidden, 0.2) for _ in range(layers)
        ]) 
        
        # classification
        self.fLinear = nn.Linear(hidden, outputs)
        
    def forward(self, embeds, swing):
        # embed model
        embeds = self.eInput(embeds)
        for block in self.eBlock:
            embeds = block(embeds)
        
        # swing traits
        swing = self.sInput(swing)
        for block in self.sBlock:
            swing = block(swing)
        
        # combine
        combined = torch.cat([swing, embeds], dim=0)
        combined = self.combine(combined)
        for block in self.backBone:
            combined = block(combined)
        
        # output 
        return self.fLinear(combined)

class bipDataset(Dataset):
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

def train(model, dataLoader, valLoader, optimizer, lossFunc, epochs):
    model.to('cuda')
    best_loss = np.inf
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for batchIdx, batchFeat in enumerate(dataLoader):
            # all feature in the batch to device
            for key, value in batchFeat.items():
                batchFeat[key] = value.to('cuda')
            # zero the gradient accumilation
            optimizer.zero_grad()
            # model pass
            predicted = model(batchFeat["embeds"], batchFeat['traits'])
            labels = batchFeat['labels'].reshape(-1, 1)
            # loss function
            loss = lossFunc(predicted, labels)
            # opti step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            # add loss value
            total_loss += loss.item()
        # train loss
        train_loss = total_loss / len(dataLoader)

        # validation loop
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for valIdx, valFeat in enumerate(valLoader):
                # all feature in the batch to device
                for key, value in valFeat.items():
                    valFeat[key] = value.to('cuda')
                # model pass
                predicted = model(valFeat["embeds"], valFeat["traits"])
                labels = valFeat['labels'].reshape(-1, 1)
                # loss function
                loss = lossFunc(predicted, labels)
                # add loss value
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valLoader)

        # save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), '../models/swingModel.pth')
        print(f'epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}')
    return model

# %% model prep