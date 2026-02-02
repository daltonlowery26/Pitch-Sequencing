# %% P(swing | pitch)
import polars as pl
import numpy as np
import torch
import gc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data')

# %% general res block
class resBlock(nn.Module):
    def __init__(self, dim, dropout):
        super(resBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.block(x) + x

# %% P(contact | swing = 1, traits, pitch embedding)
class takeMLP(nn.Module):
    def __init__(self, input, dim, layers, dropout):
        super(takeMLP, self).__init__()
        # embedding
        self.input = nn.Linear(64, dim)
        
        self.layers = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(layers)
        ])
        
        self.act = nn.BatchNorm1d(dim)
        self.lin = nn.ReLU()
        self.out = nn.Linear(dim, 1)
        

    def forward(self, embed):
        embed = self.input(embed)

        for layer in self.layers:
            embed = layer(embed)

        # post layer activation and output logits
        embed = self.act(embed)
        embed = self.lin(embed)
        embed = self.out(embed)

        return embed 

class takeDataset(Dataset):
    def __init__(self, df):
        # cols to tensor
        def col_to_tensor(col_name):
            return torch.tensor(df[col_name].to_list(), dtype=torch.float32)

        # extract from df
        pitch_embeddings = col_to_tensor('embed') # already normalized
        label = col_to_tensor('swing')

        # combined features
        self.embeds = pitch_embeddings
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
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

            predicted = model(batchFeat["embeds"])
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
                predicted = model(valFeat["embeds"])
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

# %% preparing data
df = pl.scan_parquet('cleaned_data/embed/output/pitch_umap50.parquet').select(['embed', 'swing']).collect(engine="streaming")
df = df.drop_nulls()

# %% data 
contactData = takeDataset(df)
train_size = int(0.8 * len(contactData))
val_size = int(0.1 * len(contactData))
test_size = len(contactData) - train_size - val_size

# split data
train_dataset, val_dataset, test_dataset = random_split(
    contactData,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(26)
)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# %% model
model = takeMLP(input=1, dim=512, dropout=0.2, layers=10)
loss = nn.BCEWithLogitsLoss()
opti = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
sum(p.numel() for p in model.parameters())

# %% mem chache
gc.collect()
torch.cuda.empty_cache()

# %% model train
model = train(model=model, dataLoader=train_loader, valLoader=val_loader, optimizer=opti, 
            lossFunc=loss, epochs=8)

# %% testing
def test(model, testLoader):
    model.eval()
    gLabels = []
    gPreds = []
    
    for batchIdx, batchFeat in enumerate(testLoader):
        # to cuda
        for k, v in batchFeat.items():
            batchFeat[k] = v.to('cuda')

    
        predictions = model(batchFeat["embeds"])
        labels = batchFeat['labels'].reshape(-1, 1)
        predictions = torch.sigmoid(predictions)
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        gLabels.append(labels)
        gPreds.append(predictions)

    all_predictions = np.concatenate(gPreds)
    all_labels = np.concatenate(gLabels)
    return all_predictions, all_labels

# %% load and test
stateDict = torch.load('../models/swingModel.pth')
model.load_state_dict(stateDict)
model.to('cuda')
predictions, labels = test(model, test_loader)

print(accuracy_score(labels, predictions > 0.5))
print(f1_score(labels, predictions > 0.5))
print(roc_auc_score(labels, predictions))
#k15
#0.7380146869061305
#0.7379458272121807
#0.8133362724200544

#k50
#0.7381038406494146
#0.7305207659401887
#0.8144617030084979

