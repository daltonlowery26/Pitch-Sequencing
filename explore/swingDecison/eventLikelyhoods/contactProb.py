# %% packages
import polars as pl
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

# load and select data
swing_features = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction', 'embeds']
df = (pl.scan_parquet('cleaned_data/embed/output/pitch_embeded.parquet').drop_nulls(subset=swing_features)).collect(engine="streaming")
xswing = pl.read_parquet('cleaned_data/metrics/xswing/swingTraits.parquet')
xswing = xswing.select(pl.all().name.suffix('_x'))

# %% only registered swings
df = pl.concat([df, xswing], how='horizontal')

# %% general res block
class resBlock(nn.Module):
    def __init__(self, dim, dropout):
        super(resBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

# %% P(contact | swing = 1, traits, pitch embedding)
class contactMLP(nn.Module):
    def __init__(self, input, dim, dropout):
        super(contactMLP, self).__init__()
        # embedding
        self.eInput = nn.Linear(64, dim)
        self.embedding = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(3)
        ])
        
        # pitch
        self.sInput = nn.Linear(5, dim)
        self.swing = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # output block
        self.fusion = nn.Linear(2 * dim, dim)
        
        self.output = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(3)
        ])
        
        # post act function
        self.act = nn.BatchNorm1d(dim)
        # non linear
        self.lin = nn.ReLU()
        # to output catagories
        self.out = nn.Linear(dim, 1)
    
    def forward(self, embed, swing):
        embed = self.eInput(embed)
        for layer in self.embedding:
            embed = layer(embed)
        
        swing = self.sInput(swing)
        swing = self.swing(swing)
        
        # combine swing and embedding head
        learned = torch.cat((swing, embed), dim=1)
        learned = self.fusion(learned)
        
        # from combined representation
        for layer in self.output:
            learned = layer(learned)
        
        # post res activation and output as logits    
        learned = self.act(learned)
        learned = self.lin(learned)
        learned = self.out(learned)

        return learned
class contactDataset(Dataset):
    def __init__(self, df):
        # cols to tensor
        def col_to_tensor(col_name):
            return torch.tensor(df[col_name].to_list(), dtype=torch.float32)
        
        # extract from df
        pitch_embeddings = col_to_tensor('embeds') # already normalized
        swing_traits = col_to_tensor('traits')
        label = col_to_tensor('contact')
        
        # zscore of swing traits
        mean = swing_traits.mean(dim=0)
        std = swing_traits.std(dim=0)
        swing_traits = (swing_traits - mean) / std # zscore
        
        # combined features
        self.traits = swing_traits
        self.embeds = pitch_embeddings
        self.label = label
        
    def __len__(self):
        return len(self.traits)
        
    def __getitem__(self, idx):
        return {
            "swing_traits": self.traits[idx],
            "embeds":self.embeds[idx],
            "labels": self.label[idx]
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
            predicted = model(batchFeat["embeds"], batchFeat["swing_traits"])
            # loss function
            loss = lossFunc(predicted, batchFeat["labels"])
            # opti step
            loss.backward()
            nn.utils.clip_grad_norm_(model.paramters(), max_norm=1)
            optimizer.step()
            # add loss value
            total_loss += loss.item()
        # train loss    
        train_loss = total_loss / len(dataLoader)
        
        # validation loop
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batchIdx, batchFeat in enumerate(valLoader):
                # all feature in the batch to device
                for key, value in batchFeat.items():
                    batchFeat[key] = value.to('cuda')
                # model pass
                predicted = torch.sigmoid(model(batchFeat["features"]))
                # loss function
                loss = lossFunc(predicted, batchFeat["labels"])
                # add loss value
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valLoader)
        
        # save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.dict(), '../models/contactModel.pth')
            
        print(f'epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}')
    return model

# %% preparing data
df_s = df.filter(pl.col('swing'))
df_s = df_s.with_columns(contact = (pl.col('description') == 'hit_into_play').cast(pl.Int16))
swing_traits = ['bat_speed_x', 'swing_length_x', 'swing_path_tilt_x', 'attack_angle_x', 'attack_direction_x']
df_contact_train = df_s.select(['bat_speed_x', 'swing_length_x', 'swing_path_tilt_x', 'attack_angle_x', 'attack_direction_x', 'contact', 'embeds'])
df_contact_train = df_contact_train.with_columns(
    traits = pl.concat_list(swing_traits)
)
# %% contact training
model = contactMLP(input=6, dim=128, dropout=0.1)
loss = nn.BCEWithLogitsLoss()

# optimizer
opti = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

# dataset
contactData = contactDataset(df_contact_train)

# random selection of train and val
train_size = int(0.9 * len(contactData))
val_size = len(contactData) - train_size

# split into train and val
train_dataset, val_dataset = random_split(contactData, [train_size, val_size], generator=torch.Generator().manual_seed(26))
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)

# %% train
model = train(model=model, dataLoader=train_loader, valLoader=val_loader, optimizer=opti, lossFunc=loss, epochs=10)

