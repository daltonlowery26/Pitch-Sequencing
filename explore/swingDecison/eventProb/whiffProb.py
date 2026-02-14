# %% packages
import polars as pl
import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
os.chdir('/Users/daltonlowery/Desktop/projects/Optimal Pitch/data')

# load and select data
swing_features = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction', 'embed']
df = (pl.scan_parquet('cleaned_data/embed/output/pitch_umap150.parquet').drop_nulls(subset=swing_features)).collect(engine="streaming")

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

# %% P(contact | swing = 1, traits, pitch embedding)
class contactMLP(nn.Module):
    def __init__(self, input, dim, dropout):
        super(contactMLP, self).__init__()
        # embedding
        self.eInput = nn.Linear(64, dim)
        self.embedding = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(8)
        ])

        # pitch
        self.sInput = nn.Linear(5, dim)
        self.swing = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(2)
        ])

        # output block
        self.fusion = nn.Linear(2 * dim, dim)

        self.output = nn.ModuleList([
            resBlock(dim, dropout) for _ in range(10)
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

class contactDataset(Dataset):
    def __init__(self, df):
        # cols to tensor
        def col_to_tensor(col_name):
            return torch.tensor(df[col_name].to_list(), dtype=torch.float32)

        # extract from df
        pitch_embeddings = col_to_tensor('embed') # already normalized
        swing_traits = col_to_tensor('traits')
        try:
            label = col_to_tensor('contact')
        except Exception:
            print('No Labels!')
            label = None

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

def train(model, dataLoader, valLoader, optimizer, lossFunc, mean, std, epochs):
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
            swing_traits = (batchFeat["swing_traits"] - mean) / std # zscore

            predicted = model(batchFeat["embeds"], swing_traits)
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
                mean = valFeat["swing_traits"].mean(dim=0)
                std = valFeat["swing_traits"].std(dim=0)
                vswingTraits = (valFeat["swing_traits"] - mean) / std # zscore
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
            torch.save(model.state_dict(), '../models/contactModel.pth')

        print(f'epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}')
    return model
    
def normStats(trainloader):
    n_samples = 0
    g_sum = 0.0
    g_sq_sum = 0.0

    with torch.no_grad():
        for batch in trainloader:
            traits = batch['swing_traits'].to('cuda')
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
df_s = df.filter(pl.col('swing'))
df_s = df_s.with_columns(contact = ((pl.col('description') == 'swinging_strike_blocked') | ((pl.col('description') == 'swinging_strike') )).cast(pl.Int16))
swing_traits = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']
df_contact_train = df_s.select(['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction', 'contact', 'embed'])
df_contact_train = df_contact_train.with_columns(traits = pl.concat_list(swing_traits))

# dataset
contactData = contactDataset(df_contact_train)
train_size = int(0.8 * len(contactData))
val_size = int(0.1 * len(contactData))
test_size = len(contactData) - train_size - val_size

# split data
train_dataset, val_dataset, test_dataset = random_split(
    contactData,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(26)
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

Tmean, Tstd = normStats(train_loader)

# %% model
model = contactMLP(input=6, dim=512, dropout=0.2)
loss = nn.BCEWithLogitsLoss()
opti = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
sum(p.numel() for p in model.parameters())

# %%
gc.collect()
torch.cuda.empty_cache()

# %% model train
model = train(model=model, dataLoader=train_loader, valLoader=val_loader, optimizer=opti, 
            lossFunc=loss, mean=Tmean, std=Tstd, epochs=10)

# %% testing
def test(model, testLoader, mean, std):
    model.eval()
    gLabels = []
    gPreds = []
    
    for batchIdx, batchFeat in enumerate(testLoader):
        # to cuda
        for k, v in batchFeat.items():
            batchFeat[k] = v.to('cuda')
    
        vswingTraits = (batchFeat["swing_traits"] - mean) / std # zscore
    
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
def contactPredict(df):
    # load model
    model = contactMLP(input=6, dim=512, dropout=0.2)
    stateDict = torch.load('../models/sdModels/contactModel.pth')
    model.load_state_dict(stateDict)
    model.to('mps')
    
    # dataloader
    data = contactDataset(df)
    dataLoad = DataLoader(data, batch_size=512, shuffle=False)
    
    # preds
    preds = []
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
stateDict = torch.load('../models/contactModel.pth')
model = contactMLP(input=6, dim=512, dropout=0.2)
model.load_state_dict(stateDict)
model.to('cuda')
predictions, labels = test(model, test_loader, Tmean, Tstd)


print(accuracy_score(labels, predictions > 0.4))
print(f1_score(labels, predictions > 0.4))
print(roc_auc_score(labels, predictions))
# with umap k = 15
# 0.7945076600343272
# 0.3256706579331637
# 0.7317099125219667

# umap k = 50
#0.7976860975144618
#0.39219281158091746
#0.7317299344471266

# umap mixed
#0.7963384400228848
#0.3305612436792177
#0.7282632289422329

# umap k = 150
#0.7973936812662895
#0.36671435383881734
#0.7389210765082711

# triplet
#0.7936812662894921
#0.41713957330651535
#0.7326805959973623



