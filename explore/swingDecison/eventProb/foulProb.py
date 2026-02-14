# %% packages
from torch.optim.lr_scheduler import CosineAnnealingLR
import polars as pl
import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
os.chdir('/Users/daltonlowery/Desktop/projects/Optimal Pitch/data')
device = 'mps'

# load and select data
swing_features = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction', 'intercept_x', 'intercept_y', 'embed']
df = (pl.scan_parquet('cleaned_data/metrics/xswing/xtraitContact.parquet').drop_nulls(subset=swing_features)).collect(engine="streaming")

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
        return x + self.block(x)

class foulMLP(nn.Module):
    def __init__(self, dim, layer1, layer2, layer3, dropout):
        super(foulMLP, self).__init__()
        # embedding
        self.eInput = nn.Linear(64, dim)
        self.embedding = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(layer1)
        ])

        # pitch
        self.sInput = nn.Linear(7, dim)
        self.swing = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(layer2)
        ])

        # output block
        self.fusion = nn.Linear(2 * dim, dim)

        self.output = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(layer3)
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
        for layer in self.swing:
          swing = layer(swing)

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

class foulDataset(Dataset):
    def __init__(self, df):
        # cols to tensor
        def col_to_tensor(col_name):
            return torch.tensor(df[col_name].to_list(), dtype=torch.float32, device=device)

        # extract from df
        pitch_embeddings = col_to_tensor('embed') # already normalized
        traits = col_to_tensor('traits')
        intercept = col_to_tensor('inter')
        traits = torch.cat([traits, intercept], dim=1)
        try:
            label = col_to_tensor('foul')
        except Exception:
            print('no labels!')
            label = None

        # combined features
        self.traits = traits
        self.embeds = pitch_embeddings
        self.label = label

    def __len__(self):
        return len(self.traits)

    def __getitem__(self, idx):
        return {
            "traits": self.traits[idx],
            "embeds":self.embeds[idx],
            "labels": self.label[idx]
        }

def train(model, dataLoader, valLoader, optimizer, lossFunc, mean, std, epochs):
    model.to(device)
    best_loss = np.inf
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for batchIdx, batchFeat in enumerate(dataLoader):
            # all feature in the batch to device
            for key, value in batchFeat.items():
                batchFeat[key] = value.to(device)
            # zero the gradient accumilation
            optimizer.zero_grad()
            # model pass
            swing_traits = (batchFeat["traits"] - mean) / std # zscore

            predicted = model(batchFeat["embeds"], swing_traits)
            labels = batchFeat['labels'].reshape(-1, 1)
            # loss function
            loss = lossFunc(predicted, labels)
            # opti step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            annealing.step()
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
                    valFeat[key] = value.to(device)
                # model pass
                mean = valFeat["traits"].mean(dim=0)
                std = valFeat["traits"].std(dim=0)
                vswingTraits = (valFeat["traits"] - mean) / std # zscore
                predicted = model(valFeat["embeds"], vswingTraits)
                labels = valFeat['labels'].reshape(-1, 1)
                # loss function
                loss = lossFunc(predicted, labels)
                # add loss value
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valLoader)

        # save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), '../models/sdModels/foulModel.pth')

        print(f'epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}')
    return model
    
def normStats(trainloader):
    n_samples = 0
    g_sum = 0.0
    g_sq_sum = 0.0

    with torch.no_grad():
        for batch in trainloader:
            traits = batch['traits'].to(device)
            # sum and sum squared
            g_sum += traits.sum(dim=0)
            g_sq_sum += (traits ** 2).sum(dim=0)
            # add samples
            n_samples += traits.size(0)
    # mean
    mean = g_sum / n_samples
    # std dev
    var = ((g_sq_sum / n_samples) - (mean ** 2))
    std = torch.sqrt(var)

    return mean, std

# %% preparing data
swing = pl.scan_csv('cleaned_data/pitch_2015_2026.csv').select(['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count', 'description', 'events']).collect(engine="streaming")
df = df.join(swing, on=['game_pk', 'batter_id', 'pitcher_id', 'at_bat_number', 'pitch_number', 'count'], validate='1:1' ,how='left')
# only include contact
df_s = df.filter(pl.col('description') != 'swinging_strike')
# concat swing traits to array
swing_traits = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
df_s = df_s.with_columns(traits = pl.concat_list(swing_traits))
df_s = df_s.with_columns(inter = pl.concat_list('intercept_x', 'intercept_y'))
# foul
df_s = df_s.with_columns(foul = pl.col('description') == 'foul')

# %% dataset
foulData = foulDataset(df_s)
train_size = int(0.8 * len(foulData))
val_size = int(0.1 * len(foulData))
test_size = len(foulData) - train_size - val_size

# split data
train_dataset, val_dataset, test_dataset = random_split(
    foulData,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(26)
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

Tmean, Tstd = normStats(train_loader)

# %% model
epochs = 20
model = foulMLP(dim=128, layer1= 4, layer2 = 3, layer3 = 8, dropout=0.2)
loss = nn.BCEWithLogitsLoss()
opti = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
annealing = CosineAnnealingLR(opti, T_max=epochs, eta_min=1e-6)
sum(p.numel() for p in model.parameters())

# %%
gc.collect()

# %% model train
model = train(model=model, dataLoader=train_loader, valLoader=val_loader, optimizer=opti, 
            lossFunc=loss, mean=Tmean, std=Tstd, epochs=epochs)

# %% testing
def test(model, testLoader, mean, std):
    model.eval()
    gLabels = []
    gPreds = []
    with torch.no_grad():
        for batchIdx, batchFeat in enumerate(testLoader):
            # to mps
            for k, v in batchFeat.items():
                batchFeat[k] = v.to(device)
        
            vswingTraits = (batchFeat["traits"] - mean) / std # zscore
            predictions = model(batchFeat["embeds"], vswingTraits)
            labels = batchFeat['labels'].reshape(-1, 1)
            predictions = torch.sigmoid(predictions)
            labels = labels.cpu().detach().numpy()
            predictions = predictions.cpu().detach().numpy()
            gLabels.append(labels)
            gPreds.append(predictions)

    all_predictions = np.concatenate(gPreds)
    all_labels = np.concatenate(gLabels)
    return all_predictions, all_labels

# %% predeciton function
def foulPredict(df):
    # load model
    model = foulMLP(dim=128, layer1= 4, layer2 = 3, layer3 = 8, dropout=0.2)
    stateDict = torch.load('../models/sdModels/foulModel.pth')
    model.load_state_dict(stateDict)
    model.to('mps')
    
    # dataloader
    data = foulDataset(df)
    dataLoad = DataLoader(data, batch_size=512, shuffle=False)
    
    # preds
    preds = []
    with torch.no_grad():
        for batchIdx, batchFeat in enumerate(dataLoad):
            # to mps
            for k, v in batchFeat.items():
                batchFeat[k] = v.to('mps')
    
            # predections
            predictions = model(batchFeat["embeds"])
            predictions = torch.sigmoid(predictions)
            predictions = predictions.cpu().detach().numpy()
            preds.append(predictions)

    all_predictions = np.concatenate(preds)
    return all_predictions

# %% load and test
stateDict = torch.load('../models/sdModels/foulModel.pth')
model = foulMLP(dim=128, layer1= 4, layer2 = 3, layer3 = 8, dropout=0.2)
model.load_state_dict(stateDict)
model.to(device)
predictions, labels = test(model, test_loader, Tmean, Tstd)


print(average_precision_score(labels, predictions))
print(roc_auc_score(labels, predictions))

