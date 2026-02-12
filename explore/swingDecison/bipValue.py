# %%
import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/')

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
    def __init__(self, hidden, layer1, layer2, layer3, outputs):
        super(bipModel, self).__init__()
        # embeddings
        self.eInput = nn.Linear(64, hidden)
        self.eBlock = nn.ModuleList([
            resBlock(hidden, 0.2) for _ in range(layer1)
        ])
        # swingTraits
        self.sInput = nn.Linear(7, hidden)
        self.sBlock = nn.ModuleList([
            resBlock(hidden, 0.2) for _ in range(layer2)
        ])
        # output
        self.combine = nn.Linear((hidden * 2), hidden)
        self.backBone = nn.ModuleList([
            resBlock(hidden, 0.2) for _ in range(layer3)
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
        combined = torch.cat([swing, embeds], dim=1)
        combined = self.combine(combined)
        for block in self.backBone:
            combined = block(combined)
        
        # output 
        return self.fLinear(combined)

class bipDataset(Dataset):
    def __init__(self, df):
        super(bipDataset, self).__init__()
        
        # helper function
        embeddings = torch.tensor(df['embed'].to_list(), dtype=torch.float32).to('cuda')
        swingTraits = torch.tensor(df['traits'].to_list(), dtype=torch.float32)
        
        mean = swingTraits.mean(dim=0)
        std = swingTraits.std(dim=0)
        traits = (swingTraits - mean) / std
        traits = traits.to('cuda')
        
        label = torch.tensor(df['outcome'].to_list(), dtype=torch.long).to('cuda')
        self.e = embeddings
        self.t = traits
        self.l = label
    
    def __len__(self):
        return len(self.e)
    
    # return indices and hold dataset on gpu
    def __getitem__(self, idx):
        return {
            'embeds': self.e[idx],
            'traits': self.t[idx],
            'labels': self.l[idx]
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
            labels = batchFeat['labels'].view(-1)
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
                labels = valFeat['labels'].view(-1)
                # loss function
                loss = lossFunc(predicted, labels)
                # add loss value
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valLoader)

        # save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), '../models/bipModel.pth')
        print(f'epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}')
    return model

# %% data prep
df = pl.read_parquet('cleaned_data/metrics/xswing/xtraitContact.parquet')
df = df.with_columns(
    traits = pl.concat_arr(['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction','intercept_x', 'intercept_y'])
)
des = ["hit_into_play"]
df = df.filter(pl.col('description').is_in(des))

# add events, catagories
swing = pl.scan_csv('cleaned_data/pitch_2015_2026.csv').select(['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count', 'events']).collect(engine="streaming")
df = df.join(swing, on=['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count'], validate='1:1' ,how='left')
df = df.filter(pl.col('events') != 'catcher_interf')
eventsMap = {'single':0,'double':1,'triple':2, 'home_run':3}
df = df.with_columns(
    outcome=(pl.col("events").replace_strict(eventsMap, default=4)).cast(pl.Int8)
)

# final check (should only drop one row)
df = df.drop_nulls(subset=['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 
                            'attack_direction','intercept_x', 'intercept_y', 'embed', 'events'])

# %% data loading and splitting
dataset = bipDataset(df)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(26)
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# loss, model, opti
loss = nn.CrossEntropyLoss()
model = bipModel(hidden=512, layer1=10, layer2=2, layer3=6, outputs=5)
opti = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
sum(p.numel() for p in model.parameters())

# %% train model
lastModel = train(model, dataLoader=train_loader, valLoader=val_loader, optimizer=opti, lossFunc=loss, epochs=10)

# %% testing
def test(model, testLoader):
    model.eval()
    gLabels = []
    gPreds = []
    
    for batchIdx, batchFeat in enumerate(testLoader):
        # to cuda
        for k, v in batchFeat.items():
            batchFeat[k] = v.to('cuda')
    
        predictions = model(batchFeat["embeds"], batchFeat["traits"])
        labels = batchFeat['labels'].view(-1)
        predictions = torch.argmax(predictions, dim=1)
        
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        gLabels.append(labels)
        gPreds.append(predictions)

    all_predictions = np.concatenate(gPreds)
    all_labels = np.concatenate(gLabels)
    return all_predictions, all_labels

# %% load and test
stateDict = torch.load('../models/sdModels/bipModel.pth')
model = bipModel(hidden=512, layer1=10, layer2=2, layer3=6, outputs=5)
model.load_state_dict(stateDict)
model.to('cuda')
predictions, labels = test(model, test_loader)


print(accuracy_score(labels, predictions))
print(f1_score(labels, predictions, average='weighted'))
